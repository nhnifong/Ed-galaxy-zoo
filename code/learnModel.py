"""
Learn a model of galaxies from the training data
Usage:
python code/learnModel.py /media/Loonies/galaxy-data/images_training_rev1 /media/Loonies/galaxy-data/training_solutions_rev1.csv models/output.model
"""

from __future__ import division
import sys
import csv
import os
import numpy as np
from scipy import ndimage
from time import time

train_image_dir = sys.argv[1]
train_solutions = sys.argv[2]
model_filename = sys.argv[3]


def training_examples(image_dir, solu_csv):
    csv_reader = csv.reader(open(solu_csv))
    class_names = next(csv_reader)[1:]
    solutions = {}
    for item in csv_reader:
        solutions[item[0]] = map(float,item[1:])
    image_names = os.listdir(image_dir)
    num = len(image_names)
    next_rp = time()
    print 'reading image dir'
    for i,fname in enumerate(image_names):
        if time()>next_rp:
            sys.stdout.write('\r')
            sys.stdout.write("%i out of %i images (%0.1f%%)     " % (i,num,(i/num*100)))
            sys.stdout.flush()
            next_rp = time()+0.2
        galaxyid = fname.split('.')[0]
        img = ndimage.imread(os.path.join(train_image_dir,fname))
        yield (img, solutions[galaxyid])

class GalaxyModel(object):
    def __init__(self):
        pass

    def train_examples(self, examples):
        """ learn from a list of examples of the form
        (image, classes)
        """
        print repr(examples)

    def predict(self, image):
        self.a = self.b
        return classes

model = GalaxyModel()
for n,item in enumerate(training_examples(train_image_dir, train_solutions)):
    model.train_examples([item])

## TODO
# split 'training' set into 80/20
# periodically save model
    
