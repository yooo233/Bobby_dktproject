infile = r"/Users/bobbyli/Downloads/deepknowledgetracing/Bobby_dktproject/test.log"

training = []
testing = []
other = []
trainauc = []
trainepoch = []
testauc = []
testepoch = []
line = ""


with open(infile) as f:
    f = f.readlines()

def s_break(line):
    for line in f:
        if "Training" in line:
            training.append(line)
        elif "Testing" in line:
            testing.append(line)
        else:
            other.append(line)

    for line in training:
        line = line.replace("INFO:root:Training ", "")
        line = line.split(',')
        trainauc.append(float(line[1].replace("auc: ", "")))
        trainepoch.append(int(line[3].replace("epoch: ", "")))

    for line in testing:
        line = line.replace("INFO:root:Testing ", "")
        line = line.split(',')
        testauc.append(float(line[1].replace("auc: ", "")))
        testepoch.append(int(line[3].replace("epoch: ", "")))

s_break(f)
