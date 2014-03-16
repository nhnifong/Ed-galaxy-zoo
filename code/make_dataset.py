from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import *
from pylearn2.utils.string_utils import preprocess
import matplotlib.image as mpimg
import sys, os
import numpy as np
import tables
from itertools import izip
from solu_load import load_from_csv

class galaxy_zoo_dataset(DenseDesignMatrixPyTables):
    
    def __init__(self, whichset, path=None):
        
        # here, final refers to the unlabled images from which
        # we should make predictions (images_test_rev1)
        # the train/test/valid sets come from images_training_rev1
        # bigtrain is just the whole unsplit images_traininng_rev1
        assert whichset in ['train','test','valid','final','bigtrain']
        self.whichset = whichset
        # this is the final desired shape
        # the original shape is 424, 424
        self.img_shape = (100,100,3)
        
        self.img_size = np.prod(self.img_shape)

        if path is None:
            path = '${PYLEARN2_DATA_PATH}/galaxy-data/'
        
        # load data
        path = preprocess(path)
        file_n = "{}_arrays.h5".format(os.path.join(path, "h5", whichset))
        if os.path.isfile(file_n):
            self.h5file = tables.openFile(file_n, mode='r')
            root = self.h5file.root
        else:
            print "Creating %s" % file_n
            self.first_time(whichset, path, file_n)

        # create ndim for preprocessor
        root.images.ndim = len(root.images.shape)
        if self.has_targets():
            root.targets.ndim = len(root.targets.shape)
        else:
            root.targets = None

        axes=('b', 0, 1, 'c') # not sure what this means
        view_converter = DefaultViewConverter((100, 100, 3), axes)
        super(galaxy_zoo_dataset, self).__init__(X=root.images, y=root.targets,
                                                 axes=axes)

    def __repr__(self):
        res = "h5file\n"
        im = self.h5file.root.images
        res+= "    images %r %r\n" % (im.dtype, im.shape)
        try:
            ta = self.h5file.root.targets
            res+= "    targets %r %r\n" % (ta.dtype, ta.shape)
        except AttributeError:
            pass
        return res
            
    def first_time(self, whichset, path, file_n, rng=None):
        """ Responsible for creating the desired h5 file for the first time.
        It should produce the same train/test/valid split each time given
        the same random seed"""

        if rng is None:
            rng = np.random.RandomState(47658)

        train_dir = os.path.join(path, 'images_training_rev1')
        test_dir = os.path.join(path, 'images_test_rev1')
        
        gal_ids = {}
        if whichset in ['train','test','valid','bigtrain']:
            print "Loading solution CSV file..."
            solu = os.path.join(path, 'training_solutions_rev1.csv')
            class_names, solutions = load_from_csv(solu)

            # split galaxy IDs into three sets
            gal_ids['bigtrain'] = solutions.keys()
            rng.shuffle(gal_ids)
            # sp for split point
            tot = len(gal_ids['bigtrain'])
            sp1 = int(tot * 0.80)
            sp2 = int(tot * 0.96)
            gal_ids['train'] = gal_ids['bigtrain'][:sp1]
            gal_ids['test'] = gal_ids['bigtrain'][sp1:sp2]
            gal_ids['valid'] = gal_ids['bigtrain'][sp2:]
            image_dir = train_dir
        else:
            gal_ids['final'] = [n.split('.')[0] for n in os.listdir(test_dir)]
            image_dir = test_dir

        for k,v in gal_ids.items():
            print k,len(v)
            
        num_examples = len(gal_ids[whichset])
        print 'num_examples = %i' % num_examples
        self.h5file = tables.openFile(file_n, mode='w')
        root = self.h5file.root
        images = self.h5file.createCArray(root,'images',tables.Float32Atom(),
                                     shape=(num_examples,100,100,3))
        if whichset is not 'final':
            targets = self.h5file.createCArray(root,'targets',tables.Float32Atom(),
                                          shape=(num_examples,37))

        c = 0
        n = 0
        chunksize = 100
        while n < num_examples:
            l = min(chunksize,num_examples-n)
            temp_images = np.zeros((l,100,100,3))
            temp_targets = np.zeros((l,37))
            for i in xrange(l):
                galid = gal_ids[whichset][n]
                fpath = os.path.join(image_dir,galid+'.jpg')
                x = mpimg.imread(fpath).astype(np.float32)
                x = x[162:-162, 162:-162, :]
                x /= 255
                temp_images[i] = x
                if whichset is not 'final':
                    temp_targets[i] = solutions[galid]
                n += 1
            images[n-len(temp_images):n] = temp_images
            if whichset is not 'final':
                targets[n-len(temp_targets):n] = temp_targets
            c += 0
            print '%0.2f%%' % (float(n)/num_examples*100)
                
        self.h5file.flush()

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=False, rng=None):
        if self.has_targets():
            return izip(self.h5file.root.images, self.h5file.root.targets)
        else:
            return self.h5file.root.images.__iter__

    def has_targets(self):
        return self.whichset != 'final'

    def apply_preprocessor(self, preprocessor, can_fit=False):
        preprocessor.apply(self, can_fit)

    def get_design_matrix(self):
        return self.h5file.root.images

if __name__=='__main__':
    for ws in ['train','test','valid','final','bigtrain']:
        A = galaxy_zoo_dataset(ws)
        print ws, A
        pipeline = preprocessing.Pipeline()

        # Contrast normalization
        pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))
        
        # ZCA whitening
        pipeline.items.append(preprocessing.ZCA())

        # Here we apply the preprocessing pipeline to the dataset.
        A.apply_preprocessor(preprocessor=pipeline, can_fit=True)
