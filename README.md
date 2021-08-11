# Handwriting Reognition in Pytorch

**Project Description:**<br>
For this project, we built a neural network that takes in an image of a handwritten name and outputs the name.
![img1](https://github.com/amtan20/handwriting_recognition/blob/main/sample_images/VALIDATION_10919.jpg)
![img2](https://github.com/amtan20/handwriting_recognition/blob/main/sample_images/VALIDATION_2808.jpg)

At the time of this project, there were no Pytorch notebooks published on the Kaggle site for this project, so we took an existing high performing Keras implementation and translated it into Pytorch, with a few modifications.

**Data Source:**
 https://www.kaggle.com/landlord/handwriting-recognition

**Keras Notebook that our work is based on:**
https://www.kaggle.com/samfc10/handwriting-recognition-using-crnn-in-keras

**Data Types (JPG and CSV):**
- CSVs containing JPEG filename of black and white handwriting samples
- Train-test split: divided into 331,059 training and 41,382 validation respectively
- Each row contains the true label and the label length (# of chars)
**Data Inconsistencies**:
- Illegible handwriting samples have a "NULL" label; these instances were removed
- change all labels to upper case

**Preprocessing Steps:**
**Image Sizing:**
- Fixed at 256 x64
- Larger-sized images were cropped
- Smaller-sized images were padded
- Image normalized
**Labels:**
- 30 distinct characters (26 upper-case alphabet; 4 special characters(-, ', '"', ' '))
- Set max length of labels to be 64 characters; **accounts for feeding fixed-size tensors in PyTorch**
- Create a dictionary mapping characters to a number and pad with -1 until 64 characters

## CNN_RNN Model Architecture
**Inputs:** Image, Label
**CNN Model Attributes:**.  
- 3x3 kernel size (stride 2),
- 3 downsamples (3, 32, 64, 128 channels)
- Batch Normalization Layers after conv. layers
- ReLU or Mish activation functions after Batch Normalization
- 2x2 Maxpool after conv. layers 1,2; 2x1 Maxpool after 3rd/final conv. layer
**RNN Model Attributes:**.  
- outputs of CNN fed through linear layer with (in_features=1024; output_features=64)
- bidirectinal; input_size=64; hidden_size=512
- outputs of LSTM fed through linear layer with (in_features=1024; output_features=30)
- The 30 out-features represents a probability for each character
-

## Hyperparameters
- cosine learning rate scheduler
- variable batch sizes (32, 64)
- MiSH vs ReLu activation function
- ADAM optimizer

## Loss function
- CTCLoss

## Training
- send image and labels to GPU
- take the Log SoftMax of the predicted labels
- Calculate character accuracy (compare characters the label was not -1 padded
- word accuracy (only correct if all characters within a word were correct)

## Results (20 epochs; Mish Activation; batch_size=32;
- We believe that our metric for "# of correct words" was calculated inaccurately because even as our character accuracy increased, our word accuracy stayed flat
- Our character accuracy reached 20% by epoch 15
- Our CTC losses started giving us Nans after epoch 15 and the "% correct characters" metric nosedived

## Results (20 epochs; Mish Activation; batch_size=64;


## Conclusions and Lessons Learned
**CTCLoss Function:** <br>
This was our first time using the CTCloss and in a couple of our trainin losses were "Nan". After some research, we
determined that CTCLoss is subject to numerical instability if it is not on the CPU. The reason why the "Nans" were sporadic was because
we reasoned that there were certain combinations within a batch that caused the numerical instability. <br>

**Keras vs Pytorch Translation**:<br>
Since we modeled our notebook from the top-performing notebook in Kaggle, we realized that Keras
functions vs PyTorch functions had minor inconsistencies. For example, we had to permute the dimensions in order to get the CTC loss
function to work. Additionally, the CTC Loss function in PyTorch requires y_prediction to be in the form (sequence length, batch_size, # of classes).<br>

**Learning Rate:**<br>
We experimented with fixed learning rates (0.001 and 0.0001) as well as a cosine annealing learning rate scheduler. As expected, we saw better results with the learning rate scheduler. <br>


**Activation Function:**<br>
We noticed that when our model architecture used the MiSH activation function, our training times were noticeably slower; This is because the derivatives when x <0 is harder to calculate (as opposed to ReLu which zeros out the gradient when x<0 making it faster to calculate)


**Batch Size of 32 vs 64**<br>
