#!/usr/bin/env python
import argparse
from sklearn import metrics

allowedlabels = ["A","E","N","P","R"]

parser = argparse.ArgumentParser()
parser.add_argument('inputresults',type=argparse.FileType('r'),help="The produced classified results. Format is UTTERANCENAME LABEL, where LABEL is one of the five A,E,N,P,R")
parser.add_argument('labelfile',type=argparse.FileType('r'),help="The provided labelfile for the given inputresults")
args =parser.parse_args()

labels={}
for line in args.labelfile:
	line = line.rstrip('\n').split()
	assert(line[1] in allowedlabels)
	labels[line[0]] = line[1]

x_correct = []
x_pred = []
for result in args.inputresults:
	result = result.rstrip('\n').split()
	assert(result[1] in allowedlabels)
	# Append the correct result from the labels
	x_correct.append(labels[result[0]])
	# And the result which is predicted
	x_pred.append(result[1])

precision,recall,fscore,_ = metrics.precision_recall_fscore_support(x_correct,x_pred)


avg_precision = sum(precision)/len(precision)
avg_recall = sum(recall)/len(recall)
avg_fscore = sum(fscore)/len(fscore)

print "Resulting Accuracy (AVG): %.2f RecallA(AVG): %.2f, F-Score (AVG): %.2f"%(avg_precision,avg_recall,avg_fscore)
