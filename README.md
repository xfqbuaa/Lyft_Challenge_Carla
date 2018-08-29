# The Lyft Perception Challenge

### Introduction

[The Lyft Perception Challenge](https://www.udacity.com/lyft-challenge)

### Summary
Udacity workspace unexpected idle and FCN transfer learning bug fixing cost most of GPU time. Some error fixing are shown below.  

The final model works with poor performance in the remaining 1.5 hours including training and submission.

The vehicle F score result is poor. The possible reason is less epoches 10 and the small input image shape.

Deeplab v3 model is planned but not implemented.   

### Workflow

#### Pre-process
```
def preprocess_labels(label_image):
    # Identify lane marking pixels (label is 6)
    lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()
    # Set lane marking pixels to road (label is 7)
    labels_new = label_image[:,:,0]
    labels_new[lane_marking_pixels] = 7

    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = 0
    # Return the preprocessed label image
    #print(label_image.shape)
    #print(labels_new.shape)

    overlay = np.zeros_like(label_image)
    overlay[:,:,0] = np.where(labels_new==0,1,0).astype('uint8')  
    overlay[:,:,0] += np.where(labels_new==1,1,0).astype('uint8')
    overlay[:,:,0] += np.where(labels_new==2,1,0).astype('uint8')
    overlay[:,:,0] += np.where(labels_new==3,1,0).astype('uint8')
    overlay[:,:,0] += np.where(labels_new==4,1,0).astype('uint8')
    overlay[:,:,0] += np.where(labels_new==5,1,0).astype('uint8')
    overlay[:,:,0] += np.where(labels_new==8,1,0).astype('uint8')
    overlay[:,:,0] += np.where(labels_new==9,1,0).astype('uint8')
    overlay[:,:,0] += np.where(labels_new==11,1,0).astype('uint8')
    overlay[:,:,0] += np.where(labels_new==12,1,0).astype('uint8')

    overlay[:,:,1] = np.where(labels_new==7,1,0).astype('uint8')
    overlay[:,:,2] = np.where(labels_new==10,1,0).astype('uint8')
    #print(overlay[:,:,0])

    return overlay
```

Pay more attention to different tensor shape.

Another important issue is input image shape. `image_shape = (192, 256)`
1. the input image shape is related to vgg design and should be multiply 32.
2. the final frame will be resized from this image shape.
3. the larger image shape, the more computation cost.  

#### FCN save model and freeze model

```
tf.train.write_graph(sess.graph_def, '.', 'graph.pb', as_text=False)

python -m tensorflow.python.tools.freeze_graph --input_graph ./graph.pb --input_checkpoint ./models.ckpt --output_graph ./frozen.pb --output_node_names=Softmax --input_binary=true

python -m tensorflow.python.tools.optimize_for_inference --input ./frozen.pb --output ./optimized.pb --frozen_graph=True --input_names=image_input --output_names=Softmax
```

#### Post-process

Different from Udacity provided template codes for video visualization, 3 depth image are used here.
* background  [:,:, 0]
* road including lane mark [:,:,1]
* vehicle [:,:,2]

### Workspace usage strategy

A rapid model with limited data and small input image shape is a fast solution to fix all possible error and bugs. It is really exhausted to find the wrong results after long time training.

The Udacity workspace is a powerful tool for students. It is appreciated that the workspace can be running with special command without idle automatically.

### More dataset link
[Carla dataset 1](https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180528.zip)

[Carla dataset 2](https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip)

[Carla dataset 3](https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20181305.zip)

During training, only Carla dataset 2 is used for limited GPU time.

### Error fixing

#### InvalidArgumentError
```
InvalidArgumentError (see above for traceback): NodeDef mentions attr 'dilations' not in Op<name=Conv2D; signature=input:T, filter:T -> output:T; attr=T:type,allowed=[DT_HALF, DT_FLOAT]; attr=strides:list(int); attr=use_cudnn_on_gpu:bool,default=true; attr=padding:string,allowed=["SAME", "VALID"]; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW"]>; NodeDef: conv1_1/Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/gpu:0"](Processing/concat, conv1_1/filter/read). (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
           [[Node: conv1_1/Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/gpu:0"](Processing/concat, conv1_1/filter/read)]]
```

This error is caused by tensorflow version.

Please upgrade tensorflow-gpu==1.8 or add some scripts in `preinstall_scripts.sh`.

```
sudo apt-get update
sudo apt-get install -y cuda-libraries-9-0
pip install tensorflow-gpu==1.8
```

### FailedPreconditionError
There are error information when load base_graph.pb model and sess.run().
```
FailedPreconditionError (see above for traceback): Attempting to use uninitialized value conv3_2/filter
           [[Node: conv3_2/filter/read = Identity[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"](conv3_2/filter)]]
           [[Node: conv2d_transpose_1/stack/_33 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_233_conv2d_transpose_1/stack", tensor_type=DT_INT32, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
```

Load freeze_model OK.

#### PyImport_GetModuleDict
There are error information when deal with video files:
```
Fatal Python error: PyImport_GetModuleDict: no module dictionary!

Thread 0x00007fb6567fc700 (most recent call first):
 File "/opt/conda/lib/python3.6/site-packages/tqdm/_tqdm.py", line 97 in run
 File "/opt/conda/lib/python3.6/threading.py", line 916 in _bootstrap_inner
 File "/opt/conda/lib/python3.6/threading.py", line 884 in _bootstrap

Current thread 0x00007fb749162700 (most recent call first):
 File "/opt/conda/lib/python3.6/site-packages/moviepy/video/io/VideoFileClip.py", line 116 in __del__
Aborted (core dumped)
```

The solution to PyImport_GetModuleDict Error is to upgrade moviepy module.
`pip install --upgrade moviepy`

### No Softmax out node

```
KeyError: "The name 'Softmax:0' refers to a Tensor which does not exist. The operation, 'Softmax', does not exist in the graph.
```
The error was caused by comment function `helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)`.

Please make sure the function `helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)` in main.py is not commented.

***
# Udacity FCN Project Readme
# Semantic Segmentation

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow.
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy.
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well.

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
