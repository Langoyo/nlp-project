# DO NOT IMPORT ANYTHING IN THIS FILE. You shouldn't need any external libraries.

# accuracy
#
# What percent of classifications are correct?
# 
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
# return: percent accuracy bounded between [0, 1]
#
def accuracy(true, pred):
    acc = 0
    den = 0
    for i in range(len(true)):
        if true[i]==pred[i]:
            acc+=1
        den +=1
    ## YOUR CODE STARTS HERE (~2-5 lines of code) ##
    ## YOUR CODE ENDS HERE ##
    return acc/den

# binary_f1 
#
# A method to calculate F-1 scores for a binary classification task.
# 
# args -
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
# selected_class: Boolean - the selected class the F-1 
#                 is being calculated for.
# 
# return: F-1 score between [0, 1]
#
def binary_f1(true, pred, selected_class=True):
    f1 = None
    ## YOUR CODE STARTS HERE (~10-15 lines of code) ##
    f1 = []

    for label in range(3):
        precision = 0
        recall = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(true)):
            if true[i] == label and pred[i] == label:
                tp+=1
            elif true[i] != label and pred[i] == label:
                fp+=1
            elif true[i] == label and pred[i] != label:
                fn+=1
            else:
                tn+=1
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        f1.append(2*precision*recall/(precision+recall)) 
    ## YOUR CODE ENDS HERE ##
    f1 = (f1[0] + f1[1] + f1[2])/3
    return f1

# binary_macro_f1
# 
# Averaged F-1 for all selected (true/false) clases.
#
# args -
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
#
#
def binary_macro_f1(true, pred):
    f1 = None
    ## YOUR CODE STARTS HERE (~10-15 lines of code) ##
    f1 = []

    for label in range(3):
        precision = 0
        recall = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(true)):
            if true[i] == label and pred[i] == label:
                tp+=1
            elif true[i] != label and pred[i] == label:
                fp+=1
            elif true[i] == label and pred[i] != label:
                fn+=1
            else:
                tn+=1
        precision = tp/(tp+fp+1)
        recall = tp/(tp+fn+1)

        f1.append(2*precision*recall/(precision+recall)) 
    ## YOUR CODE ENDS HERE ##
    f1 = (f1[0] + f1[1] + f1[2])/3
    return f1
