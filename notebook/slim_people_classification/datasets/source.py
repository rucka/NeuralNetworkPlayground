import glob
import os
from pandas import DataFrame, concat
from pandas.io.parsers import read_csv
import numpy as np

IMAGE_PREFIX = 'coarse_tilt_aligned_face'

def read(path):
    files=glob.glob(os.path.join(path, '*.txt'))
    fn = lambda f: read_csv(files[0], sep='\t')
    vfunc = np.vectorize(fn)
    frames=vfunc(files)
    df=concat(frames)
    df['filename'] = df.apply(lambda f: _to_filename(f, path, IMAGE_PREFIX), axis=1)
    df = df.drop(df.columns[0:3], axis=1)
    df = df.drop(df.columns[2:-1], axis=1)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    return _clean_data(df[cols])

def _to_filename(row, path, prefix_filename):
    fname = prefix_filename + '.' + str(row[2]) + '.' + row[1]
    return os.path.join(path,row[0],fname)

def _clean_data(df):
    df = df.replace('u', '').replace(np.nan, '').replace('None', '')
    #df = df[df.apply(lambda x: tf.gfile.Exists(x['filename']) , axis=1)]
    return df

def labels_gender(df):
    return np.sort(df['gender'].unique()).astype(np.str)

def labels_age(df):
    return np.sort(df['age'].unique()).astype(np.str)
