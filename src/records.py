import tensorflow as tf

class RecordProvider():
    def __init__(self, session, tfrecords, batch_size=1):
        self.session = session
        self.tfrecords = tfrecords
        self.batch_size = batch_size

    def __len__(self):
        record_num = sum(1 for _ in tf.python_io.tf_record_iterator(self.tfrecords))
        return record_num

    def feed(self):
        filename_queue = tf.train.string_input_producer([self.tfrecords])
        record_reader = tf.TFRecordReader()

        _, example = record_reader.read(filename_queue)
        features = tf.parse_single_example(example, features={
            'height' : tf.FixedLenFeature([], tf.int64),
            'width' : tf.FixedLenFeature([], tf.int64),
            'image_raw' : tf.FixedLenFeature([], tf.string)})

        img_buffer = features['image_raw']
        img = tf.image.decode_jpeg(img_buffer, channels=3)
        
        # some preprocessing
        img = tf.image.random_flip_left_right(img)
        img = tf.image.resize_images(img, [256, 256])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        img_batch = tf.train.batch([img], batch_size=self.batch_size)

        return self.session.run(img_batch)

