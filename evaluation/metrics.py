from funchub.math import custom_round
import re

def parse_answer(answer, pattern:str="####"):
    if pattern=="####":
        answer = answer.split("####")[-1]
        answer = answer.strip().strip("\n").strip('\\n')
        # 32,333 -> 32333
        answer = answer.replace(",", "")
    elif pattern=="answer is":
        answer = answer.split("answer is")[-1]
        answer = answer.strip().strip("\n").strip('\\n')
        # 32,333 -> 32333
        answer = answer.replace(",", "")

        # get the last number
        try:
            answer = re.findall(r"[-+]?\d*\.\d+|\d+", answer)[-1]
        except:
            answer = 0
    else:
        raise NotImplementedError(f"Pattern should be either #### or answer is, but got {pattern}")

    return answer

def accuracy(preds, labels, type="em"):

    if len(preds) != len(labels):
        print(f"Noticed different length of pred and label: {len(preds)} vs {len(labels)}")
        print(f"Only the first {len(preds)} elements will be evaluated")
        # only keep the first len(preds) elements
        labels = labels[:len(preds)]
        
    if type == "em":
        tuple_labels = []
        # interate through the label
        for label in labels:
            # convert it to float
            label = float(label)
            # round it to precision
            label = custom_round(label, 2)
            # check if it is an integer, i.e., 0.00, 1.00, 2.00, etc, or just 0, 1, 2, etc
            if label.is_integer():
                tuple_labels.append(("int", int(label)))
            else:
                tuple_labels.append(("float", label))
        
        correct = 0
        # check the preds
        for p, l in zip(preds, tuple_labels):
            
            try:
                # strip some characters
                p = p.strip().strip("\n").strip('\\n').strip("$")
                # convert it to float
                p = float(p)
                # round it to precision
                p = custom_round(p, 2)

                if l[0] == "int":
                    p = int(p)
                
                if p == l[1]:
                    correct += 1
                
            except:
                pass
        
        return correct / len(labels)

    elif type == "approx":
        correct = 0
        for p, l in zip(preds, labels):
            try:
                # strip some characters
                p = p.strip().strip("\n").strip('\\n').strip("$")
                # convert all the numbers to float
                p = float(p)
                l = float(l)

                # 0.1% error tolerance, e.g. 1000 -> 999 ~ 1001
                if abs(p - l) <= abs(l) * 0.001:
                    correct += 1
            except:
                pass
        
        return correct / len(labels)
    
    else:
        raise NotImplementedError(f"Accuracy type should be either em for exact match or approx for approximate match, but got {type}")



    
