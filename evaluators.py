from collections import Counter

def true_positives(determined_Y, real_Y, label):
  true_positives = 0
  for ii in range(0,len(determined_Y)):
    if determined_Y[ii] == label and real_Y[ii] == label: 
      true_positives+=1
  print "true_positives : ",true_positives
  return true_positives

def all_positives(determined_Y, label):
  return Counter(determined_Y)[label]

def false_negatives(determined_Y, real_Y, label):
  false_negatives = 0
  for ii in range(0,len(determined_Y)):
    if determined_Y[ii] != label and real_Y[ii] == label: 
      false_negatives+=1
  print "false_negatives : ",false_negatives
  return false_negatives
  
def precision(determined_Y, real_Y, label):
    if float(all_positives(determined_Y, label)) == 0: return 0
    prec=true_positives(determined_Y, real_Y, label) / float(all_positives(determined_Y, label))
    return "precision : ",prec

def recall(determined_Y, real_Y, label):
    denominator = float((true_positives(determined_Y, real_Y, label) + false_negatives(determined_Y, real_Y, label)))
    if denominator == 0: return 0
    return true_positives(determined_Y, real_Y, label) / denominator

def f1_score(determined_Y, real_Y, label = 1):
    p = precision(determined_Y, real_Y, label)
    print p
    r = recall(determined_Y, real_Y, label)
    print r
    if p[1] + r == 0: return 0
    f1 = 2 * (p[1] * r) / (p[1] + r)
    return f1
