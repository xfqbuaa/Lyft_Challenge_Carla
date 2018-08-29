import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import scipy.misc
import argparse  
import os
import scipy


file = sys.argv[-1]

image_shape = (192, 256)

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

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

def seg_function(rgb_frame):
    image = scipy.misc.imresize(rgb_frame, image_shape)
    probs = sess.run([softmax], {image_input: [image], keep_prob: 1.0})
    im_softmax = probs[0].reshape(image_shape[0], image_shape[1], 3)
    road =  np.where(im_softmax[:,:,1] > 0.5,1,0)
    road = road * 255
    car = np.where(im_softmax[:,:,2] > 0.3,1,0)
    car = car * 255
    segmentation = np.zeros_like(image)
    segmentation[:,:,1] = road
    segmentation[:,:,2] = car
    segmentation = scipy.misc.imresize(segmentation, (600,800))
        
    return segmentation

video = skvideo.io.vread(file)

GRAPH_FILE = 'frozen.pb'
graph = load_graph(GRAPH_FILE)
image_input = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
softmax = graph.get_tensor_by_name('Softmax:0')

tf.reset_default_graph()
with tf.Session(graph=graph) as sess:
    
    answer_key = {}

    # Frame numbering starts at 1
    frame = 1

    for rgb_frame in video:
	
        # Grab red channel	
	    out = seg_function(rgb_frame)    
        # Look for red cars :)
	    binary_car_result = np.where(out[:,:,2]==255,1,0).astype('uint8')
    
        # Look for road :)
	    binary_road_result =  np.where(out[:,:,1]==255,1,0).astype('uint8')

	    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
        # Increment frame
	    frame+=1

    # Print output in proper json format
    print (json.dumps(answer_key))