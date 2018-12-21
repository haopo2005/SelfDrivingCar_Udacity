#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    print("Layer image_input Shape",image_input.get_shape())
    print("Layer layer3 Shape",layer3.get_shape())
    print("Layer layer4 Shape",layer4.get_shape())
    print("Layer layer7 Shape",layer7.get_shape())
    return image_input, keep_prob, layer3, layer4, layer7
#tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    weights_initializer_stddev = 0.01
    weights_regularized_l2 = 1e-3
    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

    one_by_one_layer = tf.layers.conv2d(vgg_layer7_out, 
                                        num_classes, 
                                        1, 
                                        padding='same', 
                                        kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                        name='one_by_one_layer')
    
    transpose_layer1 = tf.layers.conv2d_transpose(one_by_one_layer, 
                                                  num_classes, 
                                                  4, 
                                                  strides=(2,2), 
                                                  padding='same',
                                                  kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2), 
                                                  name='transpose_layer1')
    
    reshape_pool4_out_scaled = tf.layers.conv2d(pool4_out_scaled, 
                                                num_classes, 
                                                1, 
                                                padding='same',
                                                kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),  
                                                name='reshape_pool4_out_scaled')
    
    transpose_layer2_input = tf.add(transpose_layer1, reshape_pool4_out_scaled, name='transpose_layer2_input')
    
    transpose_layer2 = tf.layers.conv2d_transpose(transpose_layer2_input, 
                                                  num_classes, 
                                                  4, 
                                                  strides=(2,2),
                                                  padding='same',
                                                  kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),  
                                                  name='transpose_layer2')
    
    reshape_pool3_out_scaled = tf.layers.conv2d(pool3_out_scaled, 
                                                num_classes, 
                                                1, 
                                                padding='same', 
                                                kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2), 
                                                name='reshape_pool3_out_scaled')
    
    transpose_layer3_input = tf.add(transpose_layer2, reshape_pool3_out_scaled, name='transpose_layer3_input')
    
    transpose_layer3 = tf.layers.conv2d_transpose(transpose_layer3_input, 
                                                  num_classes, 
                                                  16, 
                                                  strides=(8,8),
                                                  padding='same', 
                                                  kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2), 
                                                  name='transpose_layer3')
    return transpose_layer3
#tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer,[-1,num_classes],name='jst_logits')
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    
    return logits, training_operation, loss_operation
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, save_trg):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for images, gt_images in get_batches_fn(batch_size):
            _, loss =sess.run([train_op, cross_entropy_loss], 
                                feed_dict={input_image: images, correct_label: gt_images, keep_prob: 0.8, learning_rate: 0.0001})
            print("loss = {:3f}".format(loss))
    
        print("Saving model at Epoch:{}".format(epochs))
        
        model_runs_dir = './models'
        save_path = os.path.join(model_runs_dir, 'saved_model')
        save_path_pb = os.path.join(model_runs_dir, 'model.pb')
        
        
        saver_def = save_trg.as_saver_def()
        print(saver_def.filename_tensor_name)
        print(saver_def.restore_op_name)

        save_trg.save(sess, save_path)
        tf.train.write_graph(sess.graph_def, '.', save_path_pb, as_text=False)
        print('Saved normal at : {}'.format(save_path))
#tests.test_train_nn(train_nn)

def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        tf.saved_model.loader.load(sess, ["vgg16"], graph_file)
        graph = tf.get_default_graph()
        return graph

def run():
    num_classes = 2
    epochs = 4
    batch_size = 4
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    #model_dir = './jst/models'
    #tests.test_for_kitti_dataset(data_dir)

    correct_label = tf.placeholder(tf.int32, (None, 160, 576, num_classes))
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    '''
    graph = load_graph('/home/jst/share/project/udacity/code/advance_dl/CarND-Semantic-Segmentation-master/data/vgg')
    for op in graph.get_operations():
        print(op.name,op.values())
    '''
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3, layer4, layer7, num_classes)
        logits, training_operation, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        save_trg = tf.train.Saver(max_to_keep=5)
        train_nn(sess, epochs, batch_size, get_batches_fn, training_operation, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate, save_trg)
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        #saver = tf.train.Saver()
        #saver.save(sess, model_dir)

        # OPTIONAL: Apply the trained model to a video
        vid1 = './driving.mp4'
        voutput1='./driving_annotated.mp4' 
        if os.path.isfile(voutput1):
            os.remove(voutput1) 
        video_clip = VideoFileClip(vid1) #.subclip(0,2)
        ##pipeline(sess, logits, keep_prob, image_pl, image_file, image_shape)
        processed_video = video_clip.fl_image(lambda image: helper.pipeline(image,sess, logits, keep_prob, image_input, image_shape))
        ##lambda image: change_image(image, myparam)
        processed_video.write_videofile(voutput1, audio=False)  

if __name__ == '__main__':
    run()
