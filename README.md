
## Description
This project is a next-word-prediction task in <b>Uzbek language</b> using preprocessing text data and training multilayer recurrent neural network.

## Steps
* First of all, a file including a collection of Uzbek news/comments/posts is provided. This data is not so good and if you have a larger data, you can improve the results.
* Then data is preprocesed in data_preprocess.ipynb. After that vectors of data and labels are saved.
* Then in train_model.ipynb saved data is loaded and a multilayer recurrent neural network using LSTM layers is trained.
  After 10 epochs accuracy in training data is 0.923 and change of accuracy and loss is ploted:
  
| accuracy      | loss      |
|------------|-------------|
| <img src="https://github.com/adkhamboy/next-word-prediction/blob/main/img/accuracy.png" width="350"> | <img src="https://github.com/adkhamboy/next-word-prediction/blob/main/img/loss.png" width="350"> |

* Then in test_model.ipynb saved model is loaded and model is tested to predict next word:

<img src="https://github.com/adkhamboy/next-word-prediction/blob/main/img/result.png" width="350">

## How to Improve
* The most important problem of this project is not using big and appropriate primary data. Accuracy improves dramatically if you use more appropriate primary data.
* Different parameters as well as different layers can be tested for the neural network model and the accuracy can be improved.
