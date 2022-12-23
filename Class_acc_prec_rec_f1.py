def get_classifications(y_test, y_pred, positive_label='CONFIRMED'):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    for y_t, y_p in zip(y_test, y_pred):
        if y_t == positive_label:
            if y_p == positive_label:
                tp += 1
            else:
                fn += 1
        else:
            if y_p == positive_label:
                fp += 1
            else:
                tn += 1
    
    return tp, fn, fp, tn

def get_accuracy(tp, fn, fp, tn):
    acc = (tp + tn) / (tp + fn + fp + tn)
    return acc

def get_precision(tp, fn, fp, tn):
    precision = tp / (tp + fp)
    return precision

def get_recall(tp, fn, fp, tn):
    recall = tp / (tp + fn)
    return recall

def get_f1_score(tp, fn, fp, tn):
    precision = get_precision(tp, fn, fp, tn)
    recall = get_recall(tp, fn, fp, tn)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score
    
    
###############################################

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(name + " Accuracy: {:.3f}%".format(get_accuracy(*get_classifications(y_test, y_pred)) * 100))
    
    
    
###############################################

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(name + " F1 Score: {:.5f}".format(get_f1_score(*get_classifications(y_test, y_pred))))
