#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from lzma_detect import run_on_file_chunked, PRELUDE_STR, LzmaLlmDetector
from pathlib import Path
from itertools import chain
from math import sqrt
from junitparser import JUnitXml

MODELS = ['lzma', 'roberta', 'gptzero', 'openai']

plt.figure()

for model in MODELS:
    xml = JUnitXml.fromfile(f'{model}-report.xml')
    cases = []
    for suite in xml:
        for case in suite:
            cases.append(case)
    
    truths = []
    scores = []
    for c in cases:
        score = float(c._elem.getchildren()[0].getchildren()[0].values()[1])
        if 'human' in c.name:
            truths.append(1)
            if c.is_passed:
                scores.append(score)
            else:
                scores.append(score * -1.0)
        else:
            truths.append(-1)
            if c.is_passed:
                scores.append(score * -1.0)
            else:
                scores.append(score)

    y_true = np.array(truths)
    y_scores = np.array(scores) 

    # Compute the false positive rate (FPR), true positive rate (TPR), and threshold values
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    print(thresholds)
    # calculate the g-mean for each threshold
    # locate the index of the largest g-mean
    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.plot(fpr, tpr, lw=2, label=model.capitalize() + ': ROC curve (AUC = %0.2f)' % roc_auc)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black')#, label=model.capitalize() + ': Best @ threshold = %0.2f' % thresholds[ix])

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random classifier")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for LLM detection')
plt.legend(loc="lower right")
plt.savefig('ai_detect_roc.png')