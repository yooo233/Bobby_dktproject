import csv
import random


"""
# TODO: Learn the dataset
1. How many features do we have?
2. Understand those features?
3. What is the top5 most important feature? Feature Importance(featx) = AUC(full feature) - AUC(full feature - featx)
"""

def load_data(fileName):
    rows = []
    max_skill_num = 0
    max_num_problems = 0
    with open(fileName, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in reader:
            rows.append(row)
            #counter += 1
            #if counter == 201:
                #break
    index = 0
    print("the number of rows is " + str(len(rows)))
    tuple_rows = []
    #turn list to tuple
    while(index < len(rows)-1):
        problems_num = int(rows[index][0])
        tmp_max_skill = max(map(int, rows[index+1]))
        if(tmp_max_skill > max_skill_num):
            max_skill_num = tmp_max_skill
        if(problems_num <= 2):
            index += 3
        else:
            if problems_num > max_num_problems:
                max_num_problems = problems_num
            tup = (rows[index], rows[index+1], rows[index+2])
            tuple_rows.append(tup)
            index += 3
    #shuffle the tuple
    random.shuffle(tuple_rows)
    print("The number of students is ", len(tuple_rows))
    print("Finish reading data")
    return tuple_rows, max_num_problems, max_skill_num+1