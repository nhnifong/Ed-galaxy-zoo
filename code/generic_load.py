from __future__ import division
import numpy as np
from scipy import ndimage
import theano
import os
import sys
import csv
from time import time
from random import shuffle

def arrays_from_images(image_dir, solu_csv=None, randomize=True):
    ''' 

    :type image_dir: string
    :param image_dir: path to the directory containing images
    
    :type solu_csv: string
    :param solu_csv: path to a csv file containing the probabilities of class membership for a given example
                     if it is not provided, no class name array will be returned.
    
    returns an iterator to a list of examples.
    '''
    if solu_csv is not None:
        csv_reader = csv.reader(open(solu_csv))
    class_names = next(csv_reader)[1:]
        solutions = {}
        for item in csv_reader:
            solutions[item[0]] = map(float,item[1:])
    image_names = os.listdir(image_dir)
    if randomize:
        shuffle(image_names)
    num = len(image_names)
    next_rp = time()
    print 'reading image dir'
    for i,fname in enumerate(image_names):
        if time() > next_rp:
            sys.stdout.write('\r')
            sys.stdout.write("%i out of %i images (%0.1f%%)     " % (i,num,(i/num*100)))
            sys.stdout.flush()
            next_rp = time()+0.2
        galaxyid = fname.split('.')[0]
        img = ndimage.imread(os.path.join(image_dir,fname))
        img = img.astype(np.float32) / 255
        if solu_csv is not None:
            yield (galaxyid, img, solutions[galaxyid])
        else:
            yield (galaxyid, img)


def prepare_dataset(image_dir, solu_csv, train_prop=0.8):
    ''' Loads images and puts them in format expected by cnn
    then saves gzipped cPickled arrays

    :type image_dir: string
    :param image_dir: path to the dir of training images, all are assumed to be the same size

    :type solu_csv: string
    :param solu_csv: path to a csv of solutions (class membership probabilities)
    
    :type train_prop: float between 0 and 1
    :param train_prop: proportion of the data to make training, proportion of remaider to make test, rest is validation
    '''

    # need to iterate over arrays_from_images(image_dir,solu_csv)
    # need to reshape examples to be (3, w, h) then flatten them.
    
    n_train = int(len(fulldata) * train_prop)
    n_tv = len(fulldata) - n_train
    n_test = int(n_tv * train_prop)
    n_valid = n_tv - n_test
    assert (n_train + n_test + n_valid) == len(fulldata)
    
    train_set = fulldata[:n_train]
    testt_set = fulldata[n_train:n_train+n_test]
    valid_set = fulldata[len(fulldata)-n_valid:]
    


def load_data(dataset):
    ''' Loads the dataset from gzipped cPickled arrays from disk and puts them in shared variables

    :type dataset: string
    :param dataset: the path to the dataset. 
    '''
    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    shapes, train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # shapes is a dict containing information about the image sizes and other stuff yet to be determined.
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # rows correspond to an example.
    # target is a numpy.ndarray of 2 dimensions (a matrix)) that has the same length as
    # the number of rows in the input.
    # It should contain probabilities summing to 1 giving the class membership for each example

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables for the GPU"""
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [shapes, (train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__=="__main__":
    total = 0
    galid,img,sol = next(arrays_from_images('/media/Loonies/galaxy-data/images_training_rev1/',
                                            '/media/Loonies/galaxy-data/training_solutions_rev1.csv'))
    total += 4*(np.prod(img.shape)+len(sol))
    print total
    print len(os.listdir('/media/Loonies/galaxy-data/images_training_rev1/'))
    print ''
    print "%0.4f Mb" % (float(total) / 1024**2) 
