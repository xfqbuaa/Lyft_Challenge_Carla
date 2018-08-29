from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import scipy, argparse, sys, cv2, os

file = sys.argv[-1]

if file == 'demo.py':
    print ("Error loading video")
    quit


from graph_utils import load_graph

sess, _ = load_graph(‘frozen.pb’)
graph = sess.graph
probs = sess.run(softmax, {image_input: img, keep_prob: 1.0})


def seg_function(rgb_frame):
    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

    im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    

    return out

    
def seg_pipeline(rgb_frame):

    ## Your algorithm here to take rgb_frame and produce binary array outputs!
    
    out = seg_function(rgb_frame)
    
    # Grab cars
    car_binary_result = np.where(out==10,1,0).astype('uint8')
    car_binary_result[496:,:] = 0
    car_binary_result = car_binary_result * 255
    
    # Grab road
    road_lines = np.where((out==6),1,0).astype('uint8')
    roads = np.where((out==7),1,0).astype('uint8')
    road_binary_result = (road_lines | roads) * 255
    
    overlay = np.zeros_like(rgb_frame)
    overlay[:,:,0] = car_binary_result
    overlay[:,:,1] = road_binary_result
    
    final_frame = cv2.addWeighted(rgb_frame, 1, overlay, 0.3, 0, rgb_frame)
    
    return final_frame

# Define pathname to save the output video
output = 'segmentation_output_test.mp4'
clip1 = VideoFileClip(file)
clip = clip1.fl_image(seg_pipeline)
clip.write_videofile(output, audio=False)

