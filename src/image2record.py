#!/usr/bin/python3

import tensorflow as tf
import cv2
import glob
import os.path
import random
import argparse



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_img_lst(path_to_dir, source, target, img_type="jpeg", shuffle=True):
    """Get all image paths from `path_to_dir/{source,target}` of type `img_type` (default: jpeg).
    Returns a dictionary, which contains source/target image lists."""
    img_lst = {}
    for token in [source, target]:
        img_lst[token] = glob.glob(os.path.join(path_to_dir, token, "*.jpeg"), recursive=True)
        
        # shuffle image list
        if shuffle:
            random.shuffle(img_lst[token])

    return img_lst

def write_record(img_lst, path_to_rec, split=0.8):
    """Load all images from the image lists and save them in a TFRecord binary format. Note: We
    assume fixed size image sizes and therefore do not preprocess the images (e.g. resize and crop
    operations)."""
    for token in img_lst:
        # get length of image list
        split_index = int(len(img_lst[token]) * float(split))
        for cut in ["train", "test"]:
            # define record writer
            record_filename = os.path.join(path_to_rec, "{}_{}.tfrecords".format(cut, token))
            record_writer = tf.python_io.TFRecordWriter(record_filename)
            
            # define slice object for splitting into train/test sets
            split_slice = slice(split_index) if cut is "train" else slice(split_index, -1)

            for img_path in img_lst[token][split_slice]:
                # read image as RGB
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # get height and width
                height = img.shape[0]
                width = img.shape[1]

                # get raw image
                img_raw = img.tostring()

                # write example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height' : _int64_feature(height),
                    'width' : _int64_feature(width),
                    'image_raw' : _bytes_feature(img_raw)}))
                record_writer.write(example.SerializeToString())
            record_writer.close()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, help="specify path to the image directory")
    parser.add_argument("--source", required=True, help="specify basename of the source image directory")
    parser.add_argument("--target", required=True, help="specify basename of the target image directory")
    parser.add_argument("--save", required=True, help="specify destination path for the TFRecords")
    parser.add_argument("--split", required=True, type=float, help="specify split ratio in terms of training set  percentage")
    parser.add_argument("--shuffle", action="store_true", help="if specified, images will be shuffled")
    return parser

def main():

    # check arguments
    print("Checking arguments...")
    args = get_parser().parse_args()
    if not os.path.isdir(args.img_path):
        raise OSError('Directory "%s" does not exist' % args.img_path)
    if not os.path.isdir(os.path.join(args.img_path, args.source)):
        raise OSError('Directory "%s" does not exist' % os.path.join(args.img_path, args.source))
    if not os.path.isdir(os.path.join(args.img_path, args.target)):
        raise OSError('Directory "%s" does not exist' % os.path.join(args.img_path, args.target))
    if not os.path.isdir(args.save):
        raise OSError('Directory "%s" does not exist' % os.path.join(args.img_path, args.save))
    if not args.split > 0.0 and arg.split <= 1.0:
        raise ValueError('Traning set percentage must be within the interval (0,1]')
    print("OK")

    # setup params
    PATH_TO_DIR = args.img_path
    SOURCE = args.source
    TARGET = args.target
    SHUFFLE = args.shuffle
    SPLIT = args.split
    PATH_TO_DST = args.save
    
    # write binary train/test/target/source binary data files
    print("Starting converting images to TFRecords binary format...")
    img_lst = get_img_lst(PATH_TO_DIR, SOURCE, TARGET,shuffle=SHUFFLE)
    write_record(img_lst, PATH_TO_DST)
    print("Finished.")

if __name__ == "__main__":
    main()
