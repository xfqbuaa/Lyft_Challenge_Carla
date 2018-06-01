from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import scipy, argparse, sys, cv2, os
import tensorflow as tf
import scipy.misc


file = sys.argv[-1]

if file == 'demo.py':
    print ("Error loading video")
    quit
    
image_shape = (192, 256) 
data_dir = '../../../tmp'
vgg_path = os.path.join(data_dir, 'vgg')
num_classes = 3
output = 'segmentation_output_test.mp4'
clip1 = VideoFileClip(file)

def seg_function(rgb_frame):
    image = scipy.misc.imresize(rgb_frame, image_shape)
    probs = sess.run([softmax], {image_input: [image], keep_prob: 1.0})
    im_softmax = probs[0].reshape(image_shape[0], image_shape[1], 3)
    road =  np.where(im_softmax[:,:,1] > 0.5,1,0)
    road = road * 255
    car = np.where(im_softmax[:,:,2] > 0.5,11,0)
    car = car * 255
    segmentation = np.zeros_like(image)
    segmentation[:,:,1] = road
    segmentation[:,:,2] = car
    segmentation = scipy.misc.imresize(segmentation, (600,800))
        
    return segmentation


def seg_pipeline(rgb_frame):

    ## Your algorithm here to take rgb_frame and produce binary array outputs!
    out = seg_function(rgb_frame)
    
    # Grab cars
    #car_binary_result = np.where(out==10,1,0).astype('uint8')
    #car_binary_result[496:,:] = 0
    #car_binary_result = car_binary_result * 255
    
    # Grab road
    #road_lines = np.where((out==6),1,0).astype('uint8')
    #roads = np.where((out==7),1,0).astype('uint8')
    #road_binary_result = (road_lines | roads) * 255
    
    overlay = np.zeros_like(rgb_frame)
    overlay[:,:,1] = out[:,:,1] 
    overlay[:,:,2] = out[:,:,2] 
    overlay[496:,:,2] = 0
    
    final_frame = cv2.addWeighted(rgb_frame, 1, overlay, 0.3, 0, rgb_frame)

    return final_frame


tf.reset_default_graph()
with tf.Session() as sess:

    # restore from meta files
    saver = tf.train.import_meta_graph('models.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./')) 
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')    
    softmax = graph.get_tensor_by_name('Softmax:0')

    # Define pathname to save the output video
    clip = clip1.fl_image(seg_pipeline)
    clip.write_videofile(output, audio=False)

    # Test a image 
    #image_file = '../../../tmp/Train/CameraRGB/15.png'
    #image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
    #out = seg_pipeline(image)
    #out = seg_function(image)
    #scipy.misc.imsave(os.path.join('./', 'test.png'), np.array(out))