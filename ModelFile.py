
"""
A very simple MNIST classifier.
https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
Tensorflow official guide for building this model
https://www.tensorflow.org/get_started/mnist/pros
"""

# IMPORT MODULES
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image, ImageFilter

# choose action to train/create or test the model?
# notice: if the user only needs to testModel, please also make sure that all variables works for the target model
trainModel = True
testModel = True

# file locations
# file name for create/train model
outputModelName = "D:\HW4_model/Model3LayersSuper.ckpt"
# file names for test model
inputModelName = "D:\HW4_model/Model3LayersSuper.ckpt"
annotationAddress = "D:\sample/annotation.txt"  # annotation.txt 's directory
outputAddress = "D:\sample/output.txt"  # output.txt 's directory, to print out result
testSampleAddress = "D:\sample/examples/"  # test sample png files's dirctory

# variables
steps = 20000
batchSize = 100
convolution = (1, 1)
kennelSize = (2, 2)
maxPoll = (2, 2)
#Optimizer = AdamOptimizer
#https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
learningRate = 0.001
layer1Feature = 16
layer1Patch = 5, 5
layer2Feature = 32
layer2Patch = 5, 5
hiddenLayer = 100  # the more hiddenLayer number, the less general the model will perform
dropoffRate = 0.5  # reduce overfitting
layer3Feature = 64
layer3Patch = 5, 5

def imageprepare(argv):
    """
    This function returns the pixel values.
    The input is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    #newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    newImage = Image.new('L', (28, 28), (0)) #creates black canvas of 28x28 pixels

    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas

    tv = list(newImage.getdata()) #get pixel values
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    # tva = [(255-x)*1.0/255.0 for x in tv]  # if it is a white background, use this
    tva = [(1 - (255-x)*1.0/255.0) for x in tv]
    return tva


def createModel():

    sess = tf.InteractiveSession()

    # CREATE THE MODEL
    # interacting operations by manipulating symbolic variables
    # x is a placeholder
    # [None, 784] means any index, and 784 pixels
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # We create these Variables by giving tf.Variable the initial value of the Variable
    # in this case, we initialize both W and b as tensors full of zeros
    # Since we are going to learn W and b, it doesn't matter very much what they initially are
    # Notice that W has a shape of [784, 10]
    # because we want to multiply the 784-dimensional image vectors by it to
    # produce 10-dimensional vectors of evidence for the difference classes
    # b has a shape of [10] so we can add it to the output
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # We can now implement our model. It only takes one line to define it
    # First, we multiply x by W with the expression tf.matmul(x, W)
    # We then add b, and finally apply tf.nn.softmax
    # y = softmax(Wx + b)  // equation
    # y = tf.nn.softmax(tf.matmul(x, W) + b)  // code given by website, but less stable
    y = tf.matmul(x, W) + b

    # WEIGHT INITIALIZATION
    # will need to create more bias and weights
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    # CONVOLUTION AND POOLING
    # Our convolutions uses a stride of one and are zero padded
    # so that the output is the same size as the input
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, convolution[0], convolution[1], 1], padding='SAME')
    # Our pooling is plain old max pooling over 2x2 blocks
    # kennelSize 决定是的 sample 的大小
    # stride 决定每次 走的距离，是最终缩小的比例
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, kennelSize[0], kennelSize[1], 1], strides=[1, maxPoll[0], maxPoll[1], 1], padding='SAME')
    def max_pool_1x1(x):
        return tf.nn.max_pool(x, ksize=[1, kennelSize[0], kennelSize[1], 1], strides=[1, 1, 1, 1], padding='SAME')

    # First Convolutional Layer
    # will compute 32 features for each 5x5 patch
    # The first two dimensions are the patch size
    # the next is the number of input channels
    # the last is the number of output channels
    W_conv1 = weight_variable([layer1Patch[0], layer1Patch[1], 1, layer1Feature])
    # We will also have a bias vector with a component for each output channel
    b_conv1 = bias_variable([layer1Feature])
    # To apply the layer, we first reshape x to a 4d tensor,
    # with the second and third dimensions corresponding to image width and height,
    # and the final dimension corresponding to the number of color channels
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # We then convolve x_image with the weight tensor,
    # add the bias, apply the ReLU function, and finally max pool.
    # The max_pool_2x2 method will reduce the image size to 14x14.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    # will have 64 features for each 5x5 patch
    W_conv2 = weight_variable([layer2Patch[0], layer2Patch[1], layer1Feature, layer2Feature])
    b_conv2 = bias_variable([layer2Feature])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # THIRD LAYER
    W_conv3 = weight_variable([layer2Patch[0], layer3Patch[1], layer2Feature, layer3Feature])
    b_conv3 = bias_variable([layer3Feature])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_1x1(h_conv3)

    # Densely Connected Layer
    # the image size has been reduced to 7x7
    # we add a fully-connected layer with 1024 neurons to allow processing on the entire image
    # We reshape the tensor from the pooling layer into a batch of vectors,
    # multiply by a weight matrix, add a bias, and apply a ReLU
    W_fc1 = weight_variable([7 * 7 * layer3Feature, hiddenLayer])  # hidden layer
    b_fc1 = bias_variable([hiddenLayer])  # hidden layer

    h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * layer3Feature])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout
    # To reduce overfitting, we will apply dropout before the readout layer
    # We create a placeholder for the probability
    # that a neuron's output is kept during dropout
    # This allows us to turn dropout on during training,
    # and turn it off during testing
    # TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs
    # in addition to masking them, so dropout just works without any additional scaling
    # 默认是 丢掉 一半 , 0.5
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    # Finally, we add a layer, just like for the one layer softmax regression above
    W_fc2 = weight_variable([hiddenLayer, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    """
    Model is saved in the directory as declared at the beginning of this code
    Model file name is as modelName declared above
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    if trainModel:
        # IMPORT DATA
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        # train model
        for i in range(steps):
            batch = mnist.train.next_batch(batchSize)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) # 测的时候都保留
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropoffRate}) # train的时候丢掉一半
        # save model
        save_path = saver.save(sess, outputModelName)
        print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        print ("Model saved in file: ", save_path)

    if testModel:
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init_op)
        saver.restore(sess, inputModelName)
        prediction = tf.argmax(y_conv,1)
        prediction = tf.argmax(y_conv,1)
        f = open(annotationAddress, "r")
        output = open(outputAddress, "w")
        correct = 0
        count = 0
        for line in f:
            count += 1
            temp = line.split("\n")
            if len(temp) < 2:
                continue
            temp = temp[0].split("\t")
            if len(temp) < 2:
                continue
            image = temp[0]
            result = temp[1]
            final = testSampleAddress + image
            imvalue = imageprepare(final)
            res = prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)
            print(image + ": \t" + str(res[0]))
            output.write(image + ": \t" + str(res[0]))
            if str(res[0]) == result:
                correct += 1
        if count > 0:
            output.write("final accuracy rate: " + str(correct / count))
            print("final accuracy rate: " + str(correct / count))


createModel()
