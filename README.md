# RecognizingHandwrittenDigits_TensorFlow
This is a test of convolutional neural network using tensorflow

## Set up code
Before started, please take a look on the Python file, the user will need to set these values properly to run the Python code.
1. Line 14 - 17
These two boolean variables will tell the python file what to do:
If trainModel == True, then the Python will create / train / and save a model
If testModel == True, then the Python will test a given model on our HW test samples, which includes 195 png files
2. Line 19 - 26
As variable names suggest –
outputModelName is the directory (including file) name, where you want to save the trained model
inputModelName is the directory (including file) name, where you want to load and test model
annodatationAddress is a text file which contains name of png files, and their associated correct answers
outputAddress is the text file where you want to save the training info of your output model
testSampleAddress is the the given folder which contains 195 png files to test the model
Node: if you want to train and test your model at the same time, please make sure outputModelName is the same as inputModelName
3. Line 28 - 44
These are the variables which set up the architecture of your model, and control the training of your model, and will eventually determine the specs of your model.
I did not list all of the variables which influence your model, for a more complete description, visits the tersorflow official guide.

## Dependencies
Before you start to play with the code, you will also need to install tensorflow and PIL using pip.
1. pip install Pillow
2. pip install --upgrade tensorflow

## 
This model's final accuracy rate is 93%, the highest in my 101A Artificial Intelligence class. This model was used later in my final term project to recognizing handwritten equations.








