from __future__ import division
import numpy as np
import csv

def load_from_csv(fname):
    csv_reader = csv.reader(open(fname))
    class_names = next(csv_reader)[1:]
    solutions = {}
    for item in csv_reader:
        y = np.array(map(float,item[1:])).astype(np.float32)
        solutions[item[0]] = y
    return class_names,solutions

"""
Constraints

sum( class1 ) = 1
sum( class2 ) = val(1.2)
sum( class3 ) = val(2.2)
sum( class4 ) = val(2.2)
sum( class5 ) = val(2.2)
sum( class6 ) = 1
sum( class7 ) = val(1.1)
sum( class8 ) = val(6.1)
sum( class9 ) = val(2.1)
sum( class10 ) = val(4.1)
sum( class11 ) = val(4.1)



"""
