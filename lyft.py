import tensorflow as tf
import scipy.misc
import argparse  
import os
import scipy
import numpy as np


def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph
 
image_file = '../../../tmp/Train/CameraRGB/15.png'
image_shape = (192, 256)
image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)


GRAPH_FILE = 'frozen.pb'
graph = load_graph(GRAPH_FILE)
image_input = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
softmax = graph.get_tensor_by_name('Softmax:0')

for op in graph.get_operations():
    print(op.name)

#tf.reset_default_graph()
with tf.Session(graph=graph) as sess:
#with tf.Session() as sess:
    
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())   
    
    # restore from meta files
    """
    saver = tf.train.import_meta_graph('./models.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./')) 
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')    
    softmax = graph.get_tensor_by_name('Softmax:0')
    tf.train.write_graph(sess.graph_def, "", "test.pb", as_text=False)
    #tf.train.write_graph(graph, "", "test.pb",  as_text=False)
    """
    
    probs = sess.run([softmax], {image_input: [image], keep_prob: 1.0})
    probs = sess.run([softmax], {image_input: np.expand_dims(image, axis=0), keep_prob: 1.0})
    im_softmax = probs[0].reshape(image_shape[0], image_shape[1], 3)
    segmentation = np.zeros_like(image)
    road =  np.where(im_softmax[:,:,1] > 0.5,1,0)
    road = road * 255
    car = np.where(im_softmax[:,:,2] > 0.5,1,0)
    car = car * 255
    segmentation[:,:,1] = road
    segmentation[:,:,2] = car
    scipy.misc.imsave(os.path.join('./', 'test.png'), np.array(segmentation))
