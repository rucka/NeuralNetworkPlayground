import glob
import os
import string
import numpy as np
from PIL import Image
from pylab import *
from pandas import DataFrame, concat
from pandas.io.parsers import read_csv
import tensorflow as tf

def gender_to_num_op(v):
    def f1(): return tf.constant(1)
    def f2(): return tf.constant(2)
    def f3(): return tf.constant(3)
    def f4(): return tf.constant(0)
    return tf.case({
        tf.equal(v, tf.constant('m')): f1,
        tf.equal(v, tf.constant('f')): f2,
        tf.equal(v, tf.constant('u')): f3
        }, default=f4, exclusive=True)
def num_to_gender(v):
    options={0:'',1:'m',2:'f',3:'u'}
    return options[v]
def age_to_num_op(v):
    def fdef(): return tf.constant(-1)
    def f0(): return tf.constant(0)
    def f3(): return tf.constant(3)
    def f4(): return tf.constant(4)
    def f8(): return tf.constant(8)
    def f12(): return tf.constant(12)
    def f13(): return tf.constant(13)
    def f15(): return tf.constant(15)
    def f22(): return tf.constant(22)
    def f25(): return tf.constant(25)
    def f35(): return tf.constant(35)
    def f36(): return tf.constant(36)
    def f38(): return tf.constant(38)
    def f44(): return tf.constant(44)
    def f45(): return tf.constant(45)
    def f48(): return tf.constant(48)
    def f55(): return tf.constant(55)
    def f58(): return tf.constant(58)
    def f60(): return tf.constant(60)
    return tf.case({
        tf.equal(v, tf.constant('(0, 2)')): f0,
        tf.equal(v, tf.constant('3')): f3,
        tf.equal(v, tf.constant('(4, 6)')): f4,
        tf.equal(v, tf.constant('(8, 12)')): f8,
        tf.equal(v, tf.constant('13')): f13,
        tf.equal(v, tf.constant('(15, 20)')): f15,
        tf.equal(v, tf.constant('22')): f22,
        tf.equal(v, tf.constant('(25, 32)')): f25,
        tf.equal(v, tf.constant('35')): f35,
        tf.equal(v, tf.constant('36')): f36,
        tf.equal(v, tf.constant('(38, 43)')): f38,
        tf.equal(v, tf.constant('(38, 48)')): f44,
        tf.equal(v, tf.constant('45')): f45,
        tf.equal(v, tf.constant('(48, 53)')): f48,
        tf.equal(v, tf.constant('55')): f55,
        tf.equal(v, tf.constant('58')): f58,
        tf.equal(v, tf.constant('(60, 100)')): f60,
        }, default=fdef, exclusive=True)
def num_to_age(v):
    options={-1:None,0:'(0, 2)',3:'3',4:'(4, 6)',8:'(8, 12)',13:'13',
             15:'(15, 20)',22:'22',25:'(24, 32)',35:'35',36:'36',
             38:'(38, 43)',44:'(38, 48)',45:'45',48:'(48, 53)',55:'55',58:'58',
             60:'(60, 100)'}
    return options[v]
def image_path_op(content):
    r = tf.py_func(os.path.join, [content[0], tf.constant('*.') + content[2] + tf.constant('.') + content[1]], tf.string)
    r = tf.reshape(r, [])
    return r
def read_row(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    #user_id	original_image	face_id	age	gender	x	y	dx	dy	tilt_ang	fiducial_yaw_angle	fiducial_score
    record_defaults = [[""] for _ in range (5)]  + [[1] for _ in range (7)]
    content = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim="\t")
    data = tf.stack(image_path_op(content[0:3]))
    labels = tf.stack([age_to_num_op(content[3]), gender_to_num_op(content[4])] + content[5:10])
    return data, labels

def input_pipeline(path, batch_size, num_epochs=None, shuffle=False):
    filenames = tf.train.match_filenames_once(path)
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    d, l = read_row(filename_queue)

    capacity = 20 * batch_size
    if (shuffle == True):
        min_after_dequeue = 10 * batch_size
        d_batch, l_batch = tf.train.shuffle_batch(
            [d, l],
            batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue)
    else:
        d_batch, l_batch = tf.train.batch(
            [d, l],
            batch_size=batch_size,
            capacity=capacity)
    return d_batch, l_batch
def input_pipeline_for_single_feature(path, batch_size, image_prefix, image_dimension, feature_index, num_epochs=None, shuffle=False):
    path_batch, label_batch = input_pipeline(path, batch_size, None, False)
    data_batch = path_to_image(path_batch, os.path.dirname(path), image_prefix, image_dimension)
    label_batch = extract_feature(label_batch, feature_index)
    label_batch = tf.reshape(label_batch,[batch_size, 1])
    data_batch = tf.reshape(data_batch,[batch_size, image_dimension[0] * image_dimension[1] * 3])
    return data_batch, label_batch

def path_to_image(paths, root, image_prefix, size):
    def load_image(p):
        fp = os.path.join(root, string.replace(p, '*', image_prefix))
        return array(Image.open(fp).resize(size))
    def load_image_op(x):
        return tf.py_func(load_image, [x], tf.uint8)
    return tf.map_fn(load_image_op, paths, dtype=tf.uint8)

def extract_feature(features, index):
    def extract(p): return p[index]
    return tf.map_fn(extract, features, dtype=tf.int32)

def load_dataFrame(path, prefix_filename):
    def to_filename(row, prefix_filename):
        fname = prefix_filename + '.' + str(row[2]) + '.' + row[1]
        return os.path.join(os.path.dirname(path),row[0],fname)

    files=glob.glob(os.path.join(path, '*.txt'))
    fn = lambda f: read_csv(files[0], sep='\t')
    vfunc = np.vectorize(fn)
    frames=vfunc(files)
    df=concat(frames)
    df['filename'] = df.apply(lambda f: to_filename(f, prefix_filename), axis=1)
    df = df.drop(df.columns[[0, 1, 2]], axis=1)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    return df[cols]
