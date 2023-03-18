import logging
from typing import List

import torch
import torch.nn as nn
import argparse
import numpy as np
import time
from data import load_data
from deep_knowledge_tracing_model import DeepKnowledgeTracing
from dkt_embed_model import DKTEmbed

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from parsing import trainauc, trainepoch, testauc, testepoch

parser = argparse.ArgumentParser(description='Deep Knowledge tracing model')
parser.add_argument('-epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer')
parser.add_argument('-l2_lambda', type=float, default=0.3, help='Lambda for l2 loss')
parser.add_argument('-learning_rate', type=float, default=0.1, help='Learning rate')
parser.add_argument('-max_grad_norm', type=float, default=20, help='Clip gradients to this norm')
parser.add_argument('-keep_prob', type=float, default=0.6, help='Keep probability for dropout')
parser.add_argument('-hidden_layer_num', type=int, default=1, help='The number of hidden layers')
parser.add_argument('-hidden_size', type=int, default=200, help='The number of hidden nodes')
parser.add_argument('-evaluation_interval', type=int, default=5, help='Evalutaion and print result every x epochs')
parser.add_argument('-batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('-epochs', type=int, default=150, help='Number of epochs to train')
parser.add_argument('-allow_soft_placement', type=bool, default=True, help='Allow device soft device placement')
parser.add_argument('-log_device_placement', type=bool, default=False, help='Log placement ofops on devices')
parser.add_argument('-train_data_path', type=str, default='data/0910_all_train.csv', help='Path to the training dataset')
parser.add_argument('-test_data_path', type=str, default='data/0910_all_test.csv',help='Path to the testing dataset')
parser.add_argument('-model_type', type=int, default=1, help='Type of model')

args = parser.parse_args()
print(args)


def add_gradient_noise(t, stddev=1e-3):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """

    m = torch.zeros(t.size())
    stddev = torch.full(t.size(), stddev)
    gn = torch.normal(mean=m, std=stddev)
    return torch.add(t, gn)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def run_epoch(m, optimizer, students, batch_size, num_steps, num_skills, training=True, epoch=1):
    """Runs the model on the given data."""

    lr = args.learning_rate # learning rate
    total_loss = 0
    input_size = num_skills * 2
    print(num_skills)
    start_time = time.time()
    index = 0
    actual_labels = []
    pred_labels = []
    hidden = m.init_hidden(batch_size)
    count = 0
    batch_num = len(students) // batch_size
    while(index+batch_size < len(students)):
        x = np.zeros((batch_size, num_steps))
        target_id: List[int] = []
        target_correctness = []
        for i in range(batch_size):
            student = students[index+i]
            problem_ids = student[1]
            correctness = student[2]
            for j in range(len(problem_ids)-1):
                problem_id = int(problem_ids[j])
                label_index = 0
                if(int(correctness[j]) == 0):
                    label_index = problem_id
                else:
                    label_index = problem_id + num_skills
                x[i, j] = label_index
                target_id.append(i*num_steps*num_skills+j*num_skills+int(problem_ids[j+1]))
                target_correctness.append(int(correctness[j+1]))
                actual_labels.append(int(correctness[j+1]))

        index += batch_size
        count += 1
        target_id = torch.tensor(target_id, dtype=torch.int64)
        target_correctness = torch.tensor(target_correctness, dtype=torch.float)

        # One Hot encoding input data [batch_size, num_steps, input_size]
        x = torch.tensor(x, dtype=torch.int64)
        x = torch.unsqueeze(x, 2)
        input_data = torch.FloatTensor(batch_size, num_steps, input_size)
        input_data.zero_()
        input_data.scatter_(2, x, 1)

        if training:
            hidden = repackage_hidden(hidden)
            # hidden = m.init_hidden(batch_size)
            optimizer.zero_grad()
            # m.zero_grad()
            output, hidden = m(input_data, hidden)

            # Get prediction results from output [batch_size, num_steps, num_skills]

            output = output.contiguous().view(-1)
            logits = torch.gather(output, 0, target_id)

            # preds
            preds = torch.sigmoid(logits)
            for p in preds:
                pred_labels.append(p.item())

            # criterion = nn.CrossEntropyLoss()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, target_correctness)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(m.parameters(), args.max_grad_norm)
            # for p in m.parameters():
            #     # p.grad.data = add_gradient_noise(p.grad.data)
            #     # grad = add_gradient_noise(p.grad.data)
            #     grad = p.grad.data
            #     p.data.add_(-lr, grad)
            optimizer.step()

            total_loss += loss.item()
        else:
            with torch.no_grad():
                m.eval()
                output, hidden = m(input_data, hidden)

                output = output.contiguous().view(-1)
                logits = torch.gather(output, 0, target_id)

                # preds
                preds = torch.sigmoid(logits)
                for p in preds:
                    pred_labels.append(p.item())

                # criterion = nn.CrossEntropyLoss()
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, target_correctness)
                total_loss += loss.item()
                hidden = repackage_hidden(hidden)

        # print pred_labels
        rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
        fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("Epoch: {},  Batch {}/{} AUC: {}".format(epoch, count, batch_num, auc))

        # calculate r^2
        r2 = r2_score(actual_labels, pred_labels)

    return rmse, auc, r2

def showPlot(x, y, x2, y2, label1, label2, title):
    """
    Produce epochs vs AUC plot, and save it as a .png or .jpeg file.
    X-axis is epochs, Y-axis is AUC
    Inputs:
        - x: List[float]
        - y: List[float]
        - title: string optional
    Outputs:
        return bool
    """
    # x2, y2, title="epochs VS AUC", label1="", label2=""
    try:
        plt.figure()
        # plt.plot(x, y, '--ok')
        plt.plot(x, y, x2, y2)
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.title(title)
        plt.legend([label1, label2], loc=0)

        plt.savefig("plot.png")
        # 1. Plot
        ######### Code HERE #########
        #############################

        # 2. Rename X-axis and Y-axis


        # 3. Title

        # 4. Save as png file

        # if success return True
        print("Plot successful")
        return True
    except Exception as e:
        print("plot failed %s " % e)
        return False



def main():
    logging.basicConfig(filename='/Users/bobbyli/Downloads/deepknowledgetracing/Bobby_dktproject/1layer_default.log', level=logging.DEBUG, force=True)
    train_data_path = args.train_data_path
    test_data_path  = args.test_data_path
    choice = args.model_type
    # batch_size = args.batch_size
    batch_size = 32
    train_students, train_max_num_problems, train_max_skill_num = load_data(train_data_path)
    num_steps = train_max_num_problems
    num_skills = train_max_skill_num
    num_layers = 1
    test_students, test_max_num_problems, test_max_skill_num = load_data(test_data_path)
    #model = DeepKnowledgeTracing('LSTM', num_skills*2, args.hidden_size, num_skills, num_layers)

    if choice == 0 :
        model = DKTEmbed('LSTM', num_skills*2, args.hidden_size, num_skills, num_layers)
    elif choice == 1 :
        model = DeepKnowledgeTracing('LSTM', num_skills*2, args.hidden_size, num_skills, num_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.epsilon)
    for i in range(args.epochs):
        rmse, auc, r2 = run_epoch(model, optimizer,  train_students, batch_size, num_steps, num_skills, epoch=i)
        print(rmse, auc, r2)
        logging.info("Training rmse: {}, auc: {}, r2: {}, epoch: {}".format(rmse, auc, r2, i))

        # Testing
        if ((i + 1) % args.evaluation_interval == 0):
            rmse, auc, r2 = run_epoch(model, optimizer, test_students, batch_size, num_steps, num_skills, training=False)
            print('Testing')
            print(rmse, auc, r2)
            logging.info("Testing rmse: {}, auc: {}, r2: {}, epoch: {}".format(rmse, auc, r2, i))


if __name__ == '__main__':
    main()

    # x2 = [1, 2, 3, 4, 5, 6]
    # y2 = [6, 3, 2, 8, 9, 1]
    # showPlot(trainepoch, trainauc)
    # showPlot(trainepoch, trainauc, testepoch, testauc, "Training", "Testing", "Epoch vs. AUC for 3-layer + embedding")
    # showPlot(x, y, x2, y2, title="test", label1="test2", label2="test3")