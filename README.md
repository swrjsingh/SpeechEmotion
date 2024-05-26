## Abstract
Speech Emotion Recognition (SER) involves identifying human emotions and affective states from speech. This process leverages the fact that emotions often manifest in vocal tone and pitch. Recently, emotion recognition has become a rapidly expanding research field. While machines inherently lack the capability to perceive and express emotions, incorporating automated emotion recognition into human-computer interactions can enhance user experience by reducing the need for human intervention.

In this project, we analyze basic emotions such as calm, happy, fearful, and disgusted from emotional speech signals. We employ machine learning techniques, including the Multilayer Perceptron Classifier (MLP Classifier), which categorizes data into non-linearly separated groups. Additionally, we use Convolutional Neural Networks (CNN) and Recurrent Neural Networks with Long Short-Term Memory (RNN-LSTM) models. Features such as Mel-frequency cepstrum coefficients (MFCC), chroma, and mel are extracted from the speech signals to train the MLP classifier. To achieve this, we utilize Python libraries like Librosa, sklearn, pyaudio, numpy, and soundfile for analyzing speech modulations and recognizing emotions.

The RAVDESS dataset, which includes approximately 1500 audio files from 24 actors (12 male and 12 female) expressing 8 different emotions, will be used to train an NLP-based model. This model will be capable of detecting the 8 basic emotions and identifying the speaker's gender. Once trained, the model can be deployed to predict emotions from live voice inputs.

## Deliverables

- Learn the Basics: Gain foundational knowledge in Python, Machine Learning (ML), Deep Learning (DL), Natural Language Processing (NLP), and libraries such as Librosa and sklearn.
- Literature Review: Conduct a comprehensive review of existing research and methodologies in the field of Speech Emotion Recognition.
- Dataset Analysis and Feature Extraction: Analyze the dataset and extract relevant features.
- Model Building and Training: Build and train the model using the training data.
- Testing: Test the model on the test data.
- Live Audio Testing: Test the model on live (unseen) audio inputs and collect the results.

## Dataset

For our project, we utilized the RAVDESS dataset. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) comprises 7356 files, totaling 24.8 GB. This database features 24 professional actors (12 female, 12 male) vocalizing two lexically-matched statements in a neutral North American accent.

## Observations

During the project, we encountered several challenges, primarily in selecting the ideal features for the models and tuning the hyperparameters. After experimenting with various features, we found that Mel-frequency cepstral coefficients (MFCCs) and spectral features yielded the highest accuracy in the MLP and CNN models. Given that the LSTM model performs best with sequential data, we limited its input features to MFCCs.

All three models exhibited significant overfitting, which considerably reduced test accuracy. To mitigate this, we incorporated techniques such as dropout and regularization. We also conducted extensive efforts to optimize parameters, including the number of hidden layers, activation functions, optimizers, batch size, and test size, to maximize accuracy.

Ultimately, the final comparison of the three models showed that the CNN model delivered the best results on the dataset.

## Feature Extraction

Feature extraction is important in modeling because it converts audio files into a format that can be understood by models.

1. MFCC (Mel-Frequency Cepstral Coefficients)- It is a representation of the short-term power spectrum of a sound, based on linear cosine transformation of a log power spectrum on nonlinear mel frequency scale.
2. Chroma- It closely relates to the 12 different pitch classes. It captures harmonic and melodic characteristics of music.
3. Mel Scale- It is a perceptual scale of pitches judged by listeners to be in equal in distance from one another. 
4. Zero Crossing Rate (ZCR)- It is the rate at which a signal changes from positive to zero to negative or from negative to zero to positive.
5. Spectral Centroid- It is the center of 'gravity' of the spectrum. It is a measure used in digital signal processing to characterize a spectrum.

## Implementation Details

### Objective
- Classify sound files into gender and emotion labels.
- Two separate models: one for gender classification, one for emotion classification.
- Emotion model: 8 labels (0-7).
- Gender model: 2 labels (0-1).

### Taking Inputs
- **Dataset**: 1440 files, each ~3 seconds long.
- **Sampling Rate**: 22100 Hz.
- **Issue**: Varying file lengths due to micro/millisecond differences.
- **Solution**: 
  - Pad shorter files with zeros to a length of 120,000 samples.
  - Files stored in a numpy array of shape (24*60, 120,000).
- **Label Extraction**:
  - Emotion labels from 6th and 7th characters of file names.
  - Labels adjusted from 1-8 to 0-7 for compatibility with softmax function in MLP classifier.

###  Feature Extraction

#### MLP Model
- Sound files converted to features before input to MLP model.
- Features: MFCC (40), chroma frequency (12), mel frequency (128), and spectral rolloff (1).
- Mean of each feature taken across all frames of a file.
- Spectral rolloff discarded due to low influence on accuracy.

#### LSTM Model
- Selected features: 20 MFCC features per frame.
- Other features negatively impacted performance and were discarded.

#### CNN Model
- Extensive experimentation to choose best features.
- Selected features: 20 MFCC features per frame, spectral centroid, bandwidth, contrast, and flatness.
- Other combinations discarded due to lower performance.


### Model Implementation Architecture

#### MLP Model

A fully connected neural network was deployed to predict emotions from the features of the sound files. 

- This model consists of four hidden layers with dropout applied to the first three layers to prevent overfitting. 
- The first and second hidden layers each have 512 neurons, the third layer has 128 neurons, and the fourth layer has 64 neurons. 
- The model was constructed using Keras's Sequential module, which allows layer-wise input specification. 
- The 'relu' activation function was used to propagate values from one layer to another. 
- Dropout of 0.1 was added to the first three hidden layers to randomly drop nodes with a 10% probability, helping to prevent overfitting.
- Additionally, L2 regularization was applied to the first two layers to further mitigate overfitting and fine-tune the parameters. The input layer has a shape of (180, 1), and the output layer uses the 'softmax' function to predict the emotion by providing the probability for each label. The label with the highest probability is selected as the predicted emotion.

#### LSTM Model

A Long Short-Term Memory (LSTM) model was also implemented. 
- This model consists of five layers: three LSTM layers followed by two fully connected dense layers. 
- The LSTM layers have 256, 128, and 32 units, respectively, each with a dropout and recurrent dropout of 0.2. Kernel regularizers were added to each layer to minimize overfitting. 
- Batch normalization was applied three times to accelerate training and improve model generalization. 
- A flatten layer was used to transition from LSTM to dense layers, followed by a dense layer with 256 neurons and 'relu' activation. 
- The final layer uses the 'softmax' activation function to predict one of the eight emotions.

#### CNN Model

A Convolutional Neural Network (CNN) was trained and tested on the dataset. 
- The architecture consists of five layers: the first two are 1D convolutional layers, and the last three are fully connected dense layers. 
- The convolutional layers each have 32 filters of size 3 and use the 'relu' activation function. 
- Batch normalization was included to speed up training and improve generalization on unseen data. 
- Two dropout layers were added between the convolutional layers to enhance model performance. 
- A flatten layer followed the convolutional layers, transitioning into three fully connected layers with 512, 512, and 256 neurons, respectively, all using the 'relu' activation function. 
- Regularizers and dropout layers were also applied between these layers to prevent overfitting. The final dense layer uses the 'softmax' activation function to predict emotions.

### Model Training and Prediction
After extracting features, the data was split into training and testing sets with a test size of 0.2. The data was scaled between 0 and 1 to improve model performance. The models were then trained using this data. Sparse categorical cross-entropy was used as the loss function, and the Adam optimizer was employed during the final compilation of the trained models. After training, the models were tested on the test dataset to evaluate their performance.

## Results:

### Emotion Recognition
1. MLP Model
Accuracy obtained on training data is 96.2%
Accuracy obtained on test data is 71%

2. LSTM Model
Accuracy obtained on training data is 100 %
Accuracy obtained on test data is 75%

3. CNN Model
Accuracy obtained on training data is 98.2%
Accuracy obtained on test data is 77%

### 7.0.2 Gender Recognition
Accuracy obtained on training data is 100%
Accuracy obtained on test data is 99.4%

### On Live Voice Input
  - CNN model gave an accuracy of 73% 
  - LSTM model gave an accuracy of 71%
  - MLP model gave an accuracy of 62%



## References
1. https://towardsdatascience.com/multi-layer-perceptron-using-tensorflow-9f3e218a4809
2. https://www.researchgate.net/publication/341922737_Multimodal_speech_emotion_recognition_and_classification_using_convolutional_neural_network_techniques
3. https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
4. https://librosa.org/doc/main/tutorial.html
5. http://www.jcreview.com/fulltext/197-1594073480.pdf?1625291827
     
