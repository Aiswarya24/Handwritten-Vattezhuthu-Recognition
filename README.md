## Objective

The primary goal of our research is to build a solid deep learning model that accurately recognizes handwritten Vattezhuthu script. This goal fits in with the larger mission to save and expose the rich historical and cultural treasures concealed inside this old script. The core of our effort is handwritten character recognition since it is the key to decoding and comprehending Vattezhuthu manuscripts, inscriptions, and documents.

## Methodology

### Data Creation

We have learned the alphabets of Vattezhuthu script in the 19th century and created a dataset with 50 images in each letter folder using python code and Paint. Each letter folder is named according to their pronunciation. There are 29 letters in Vattezhuthu script consisting of 18 consonants and 11 vowels. As there are variations in the form of letters in various time periods, we have selected the letters during the 19th century to train our model.

### Data Preprocessing

### Data Augmentation

In order to build a robust model a dataset with larger training data should be used. Since our dataset is too small we preferred to perform data augmentation. We have used parameters like rotation range, shear range, zoom range, brightness range and generated a dataset with 1950 copies of each letter. These parameters define the types and degrees of transformations that will be applied to the images. We specified the directory where the images are located. Loop through each subfolder (representing letters) in the main image directory. Apply data augmentation to each  original image to generate 40 new images. This is done by fixing the batch size as 40. The code generates 40 augmented images for each of the 50 original images, resulting in a total of 50*40 = 2000 images in the same directory where the original images are located.

### Reading, Resizing and Analyzing the dataset

Reading the whole dataset using keras.preprocessing. The image size is converted to (32,32) and a batch of size 32 is created. Then in the analyzing stage each of the folder names is converted to the corresponding lower case letters. In order to get the right pronunciation of each letter this is to be done.

### Train Test Split

Splitted the dataset into three partitions: a training set, validation set and a test set. These partitions are defined by the specified split ratios. The dataset is shuffled to ensure randomness in the data. The shuffle_size parameter controls the number of elements to sample when shuffling. Seed=12 is used to initialize the random number generator before shuffling the dataset. This ensures that if someone else runs the same code with the same seed value we will get the same shuffled order of elements. Then we created the training set. After that the validation set is created by skipping the dataset that is already allocated for training purposes. Finally  the test set is also created skipping both the dataset that has already been taken by the training and validation set. Thus we get all the three training, validation and test sets.

### Caching and Prefetching
  
These transformations are used to improve the performance during training. Caching reduces data loading time, shuffling introduces randomness to the order of data samples, and prefetching overlaps data preprocessing and model training to keep hardware resources busy and improve training throughput. Similar transformations are applied to the validation and test datasets to ensure efficiency during evaluation.

### Rescaling (Normalization)

Here we resizes the input images to a target size of 32*32 pixels. This operation changes the dimensions of the images, making them smaller in this case. rescaling is a form of normalization to ensure that the pixel values are within a small range. Here we effectively transform the pixel values from the range [0,255] to the range [0,1] . This type of preprocessing is common when preparing image data for use with deep learning models, as it helps standardize the input data and can improve model training and convergence.

### Model building

The general Convolution Neural Networks model is used here to recognize the letters of vattezhuthu scripts. Our input data has a shape of (32, 32,32,3) which implies that the first dimension represents the batch size which is 32. The second and third dimension represents the height(32) and width(32) of the image respectively. Fourth dimension represents the number of color channels in each image; here it is 3 ( 3 for RGB).
The code sets up early stopping, a technique used during training to prevent overfitting and save time. The model’s validation accuracy will be monitored during training. Training will stop if the validation accuracy stops improving. Patience=5 implies that the validation accuracy does not improve for 5 consecutive epochs, training will stop early.Mode = auto implies that the monitoring criterion(here it is validation accuracy) will automatically determine whether to minimize or maximize(here it will maximize the accuracy). The min_delta parameter specifies the smallest improvement in the monitored metric(validation accuracy) that is considered significant. If the accuracy increases by less than 0.005, it is not considered an improvement, and early stopping won’t be triggered.The verbose parameter determines how much information about early stopping is displayed during training. A value of 1 means it will provide updates and messages regarding when and why training stops early based on the specified criteria. Then we do the resize and rescale which we have defined earlier.

The first convolution layer has 32 filters, a 3*3 kernel and ReLU activation, the input shape= (32,32,3)  argument in this layer specifies that this layer expects the  input image to have a shape of 32 pixels in height, 32 pixels in width and 3 color channels(RGB). Next we give a max-pooling layer with a pool size of 2*2, which reduces the spatial dimension of the data. Then we defined another convolution layer with 64 filters, 3*3 kernel and ReLU activation which is followed by another max-pooling layer with 2*2 pool size. Next layer flattens the output from the previous layers, turning it into a one dimensional vector. Next one is a fully connected (dense) layer with 64 units and ReLU activation followed by the final dense layer with 29 units and softmax activation. This suggests that the model is designed for multiclass classification into 29 classes.

### Summarizing the neural network model

Model name: Sequential_1 , This is the model name which is automatically generated by TensorFlow/ Keras. The model appears to have a total of 6 layers. This model is a neural network with fully connected and convolutional layers created for a particular goal. It may be used for a classification task with 29 classes since it accepts an input of shape (32, 32, 32, 3), and it outputs shape (32, 29). There are 168,797 trainable parameters in the model as a whole.

### Compiling the model

Compilation is used to configure the training process for a neural network model. It sets the optimizer as 'adam'. The optimizer is responsible for updating the model's weights during training to minimize the loss function. Adam is a popular optimization algorithm that adapts the learning rate during training for faster convergence. The model will use 'SparseCategoricalCrossentropy' as the loss function during training. It measures how well the model's predictions match the actual target values. It defines accuracy as the evaluation metrics. Accuracy measures the percentage of correctly classified examples in the training data.

### Training the model using .fit() function

.fit() function is used to train a neural network model.It takes input data, labels, and various training parameters to optimize the model's weights and make it better at its task. The training dataset is given to train the model. The model learns from this dataset during training. Batch_size is set as 32 to determine how many examples are used in each update of the model's weights during training. The model is evaluated on a validation dataset after each epoch (training cycle). It helps to monitor the model's performance on unseen data and detect overfitting. Training for more epochs allows the model to learn better but may lead to overfitting if not controlled. Callbacks are functions that are called at specific points during training. Early stopping monitors the validation loss and stops training when it detects that the model's performance on the validation data is no longer improving. This helps prevent overfitting and saves training time.

## model Evaluation

Here we are evaluating a trained machine learning model on a test dataset to see how well it performs on new, unseen data.
 
●	The scores variable will contain measurements like loss and accuracy, telling us how good our model is at making predictions. 

●	The history variable holds information about how our model learned during training, such as how its performance changed over time.

●	The history.params likely keeps track of training settings like batch size and the number of training cycles. 

●	Lastly, history.history.keys() gives us the names of specific training measurements, which can be helpful for analyzing how well our model is learning and improving. 

### Analyzing the accuracy and loss of training and validation dataset using the model built

acc and val_acc represent the model's training accuracy and validation accuracy over time, respectively. Similarly, loss and val_loss stand for training loss and validation loss. These metrics help us see how well our machine learning model is learning and how it generalizes to new data. Improvements in accuracy and reductions in loss indicate that the model is getting better at its task during training, which is what we typically aim for in machine learning.

### Visualization

We created two side-by-side plots to visualize the performance of a machine learning model during training. The left plot displays the training accuracy (how well the model predicts the training data) and the validation accuracy (how well it predicts new, unseen data) over ten training cycles.
 The right plot shows the training loss (a measure of how wrong the model's predictions are on the training data) and the validation loss (similar, but on new data). By comparing these curves, we can assess how well our model is learning and if it's overfitting (performing well on training but not on validation). The goal is to see improvements in accuracy and reductions in loss over time.

The graph below shows that the training accuracy approximately coincides with the validation accuracy. That is our model is free from both overfitting and underfitting errors. Similarly the training loss and validation loss degrades by each developing epoch and they more or less overlap in the final stage. 

##Saving the model

It is crucial to highlight the process for preserving the model once our Convolutional Neural Network (CNN) model was successfully trained for handwritten Vattezhuthu detection. To guarantee interoperability across different platforms and contexts, we utilized the well-known TensorFlow framework to save our trained CNN model in the SavedModel format. In addition to preserving the model's architecture and learnt weights, saving the model makes it easier to utilize and deploy it in the future. Additionally, we set up a checkpointing mechanism to preserve model states periodically throughout training, allowing us to resume where we left off in case of disruptions or to choose the model with the greatest validation metrics. The stored model will be a useful tool for further improvements, application deployment, and additional study into handwritten Vattezhuthu recognition.

## Results

The creation of a reliable model to recognise the Vattezhuthu script has been accomplished successfully, and we are pleased to share this. A training accuracy of 0.9964 and a validation accuracy of 0.9910  have been attained by this model, demonstrating its impressive performance. The model's ability to precisely differentiate and categorize the Vattezhuthu script characters is shown by these remarkable accuracy metrics, showing the model's skill in character identification. 

Furthermore, in addition to properly predicting the labels of the test data, our model also delivers the right pronunciation for each character. By introducing this functionality, the model becomes more practical and helpful, making it a crucial resource for learning and using the Vattezhuthu script. Our model's ability to recognise characters while additionally guiding users in the proper articulation and application of the script is made possible by the combination of high accuracy and pronunciation guidance. This accomplishment confirms the potential for using technology to connect old scripts with modern applications and represents an enormous advance in the field of script recognition.

   
