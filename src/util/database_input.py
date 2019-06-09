
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

     
def read_database(filename_queue,parameters):
    """Reads and parses examples from CIFAR10 data files.
    
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.
    
    Args:
      filename_queue: A queue of strings with the filenames to read from.
    
    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result 
        width: number of columns in the result 
        depth: number of color channels in the result
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0,1,2,3..n.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass
    
    result = CIFAR10Record()
    
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = parameters.IMAGE_SIZE1
    result.width = parameters.IMAGE_SIZE2
    result.depth = parameters.NUM_CHANNELS
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes
    
    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)
    
    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    
    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    
    return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(parameters):
    """Construct distorted input for CIFAR training using the Reader ops.
    
    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    
    # Create a queue that produces the filenames to read.
    if(parameters.ID_CHANNEL == 0): 
        filename_queue = tf.train.string_input_producer([parameters.PATH_OUTPUT + 'label_images.bin'])
    else:
        filename_queue = tf.train.string_input_producer([parameters.PATH_OUTPUT + 'label_images_' + str(parameters.ID_CHANNEL) + '.bin'])
        
    # Read examples from files in the filename queue.
    read_input = read_database(filename_queue,parameters)
    distorted_image = tf.cast(read_input.uint8image, tf.float32)
    
    #height = parameters.IMAGE_SIZE1
    #width = parameters.IMAGE_SIZE2
    
    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    
    # Randomly crop a [height, width] section of the image.
    #distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    
    # Randomly flip the image horizontally.
    #distorted_image = tf.image.random_flip_left_right(distorted_image)
    
    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    #distorted_image = tf.image.random_brightness(distorted_image,
    #                                             max_delta=63)
    #distorted_image = tf.image.random_contrast(distorted_image,
    #                                           lower=0.2, upper=1.8)
    
    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_standardization(distorted_image)
    
    
    
    #float_image = distorted_image
    
    
    #input_tensor = tf.placeholder(tf.float32, shape=(None,parameters.NEW_IMAGE_SIZE1,parameters.NEW_IMAGE_SIZE2,3), name='input_image')
    scaled_input_tensor = tf.scalar_mul((1.0/255), distorted_image)
    scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
    float_image = tf.multiply(scaled_input_tensor, 2.0)
    
    
    
    
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 1 ###########################
    min_queue_examples = int(parameters.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    #print ('Filling queue with %d CIFAR images before starting to train. '
    #       'This will take a few minutes.' % min_queue_examples)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, parameters.BATCH_SIZE,
                                           shuffle=True)

'''
def inputs(parameters):
    """Construct input for CIFAR evaluation using the Reader ops.
    
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [parameters.PATH_TEST]
    num_examples_per_epoch = parameters.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    
    # Read examples from files in the filename queue.
    read_input = read_database(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = parameters.IMAGE_SIZE1
    width = parameters.IMAGE_SIZE2
    
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 1   ############################### ##### mudar para 1
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, parameters.BATCH_SIZE,
                                           shuffle=False)
'''