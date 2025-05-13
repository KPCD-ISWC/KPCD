import json
from load_data import FewShotDataset

def get_result_with_comma(dataname, labels, outputs, label_set, test_data):
    assert len(labels) == len(outputs)
    TP, FP, FN = 0, 0, 0
    result = []
    total_hamming_loss = 0
    for i in range(len(labels)):
        preds = [j.strip().lower() for j in outputs[i].strip().replace('，', ',').split(',')]
        labs = [j.strip().lower() for j in labels[i]]
        tp = len(set(preds) & set(labs))
        TP += tp
        FP += len(set(preds)) - tp
        FN += len(set(labs)) - tp
        
        n_labels = len(label_set)
        pred_set = set(preds)
        lab_set = set(labs)
        n_mismatches = len(pred_set.symmetric_difference(lab_set))
        total_hamming_loss += n_mismatches / n_labels
        
        result.append(test_data[i]+'\n')
        result.append(', '.join(preds)+'\n')
        result.append(', '.join(labs)+'\n')
        result.append(test_data[i]+'\n')
        result.append(', '.join(preds)+'\n')
        result.append(', '.join(labs)+'\n')
    
    if (TP+FP) == 0:
        P = 0
    else:
        P = TP/(TP+FP)
    if (TP+FN) == 0:
        R = 0
    else:
        R = TP/(TP+FN)
    if (P+R) == 0:
        F1 = 0
    else:
        F1 = 2*P*R/(P+R)

    hamming_loss = total_hamming_loss / len(labels)
    
    return P, R, F1, hamming_loss, result

def get_result_with_str_match(dataname, labels, outputs, label_set, test_data):
    assert len(labels) == len(outputs)
    label_set = [i.strip().lower() for i in label_set]
    #print(label_set)
    TP, FP, FN = 0, 0, 0
    result = []
    total_hamming_loss = 0
    
    for i in range(len(labels)):
        preds = []
        outputs[i] = outputs[i].lower()
        #print(outputs[i])
        for c in label_set:
            if c in outputs[i]:
                preds.append(c)
        labs = [j.strip().lower() for j in labels[i]]
        tp = len(set(preds) & set(labs))
        TP += tp
        FP += len(set(preds)) - tp
        FN += len(set(labs)) - tp
        
        n_labels = len(label_set)
        pred_set = set(preds)
        lab_set = set(labs)
        n_mismatches = len(pred_set.symmetric_difference(lab_set))
        total_hamming_loss += n_mismatches / n_labels
        
        result.append(test_data[i]+'\n')
        result.append(', '.join(preds)+'\n')
        result.append(', '.join(labs)+'\n')
    
    if (TP+FP) == 0:
        P = 0
    else:
        P = TP/(TP+FP)
    if (TP+FN) == 0:
        R = 0
    else:
        R = TP/(TP+FN)
    if (P+R) == 0:
        F1 = 0
    else:
        F1 = 2*P*R/(P+R)
        
    hamming_loss = total_hamming_loss / len(labels)
    
    return P, R, F1, hamming_loss, result
