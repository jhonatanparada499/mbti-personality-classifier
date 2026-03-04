# SP26 Jhonathan Parada
ML research with Jhonathan Parada  
Currently working on: Gender classification from text using N-GRAM and BAG-OF-WORDS.
 

[Working With Text Data](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html) (reading: Training a classifier) Covers: Bags of words, n-grams, and how it relates to CountVectorizer  
|  
[NLP: Text Vectorization Methods with SciKit Learn](https://admantium.medium.com/nlp-text-vectorization-methods-with-scikit-learn-4ada4e845a73)  
|  
[Google ML Concepts/Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings) (Done)  
|  
[N-grams in NLP](https://medium.com/@abhishekjainindore24/n-grams-in-nlp-a7c05c1aff12) (Done)  
|  
classifying-user-gender-based-on-tweet-text.ipynb  

## Personality

### Datasets
- [MBTI Personality Types 500 Dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset)  

### Papers
- [Personality Detection using XLM-ROBERTa and
Whisper](papers/Personality_Detection_Using_Xlm-Roberta_and_Whisper.pdf)  

**Notes:** Paper seems to be a pipeline design to process either text or audio data(bimodal) using Whisper(ASR, whisper-large-v3), a tokenizer, and a "tuned" classification model called XLMRobertaClassifier to predict MBTI types. They used MBTI types as labels and forum posts(video links and text concatenated by |||) from profiles as features. (03-28).  

How does hybrid resampling work? Specially how did they convert less than 250 samples into 800 samples in Fig. 3 and 4?  
What do you mean by "implemented via the Hugging Face 'pipeline'" and "Hugging Face 'Trainer' API"? (03-02)

### MBTI Examples
- [Personality Prediction Project using ML](https://www.geeksforgeeks.org/machine-learning/overview-of-personality-prediction-project-using-ml/)
- [predicting personality from social media text.](https://rismakov.com/mbti-prediction/category/Scikit-learn)

### Toy Examples
#### Natural Language Processing
- Inspiration: [Classifying User Gender Based on Tweet Text](https://www.kaggle.com/code/kinguistics/classifying-user-gender-based-on-tweet-text/notebook)

#### Computer Vision
- Recognizing Written Digits Using a [Dataset](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download) from Kaggle

[recognizing_written_digits.ipynb](toy_examples/comp_vision/recon_written_digits/recognizing_written_digits.ipynb) is trained using half the [kaggle_written_digits](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download) dataset from Kaggle. Then, it is tested (validated) using its next half. Furthermore, to prove compatibility with other datasets, the [Scikit-learn digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) dataset from Scikit-learn is passed to the model to make predictions. The performance metrics for both cases are:

Results on [kaggle_written_digits](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download) test partiton
```
Classification report for classifier SVC(gamma=0.001):
              precision    recall  f1-score   support

           0       0.92      0.92      0.92      1138
           1       0.85      0.94      0.89      1126
           2       0.90      0.89      0.90      1101
           3       0.80      0.83      0.81      1092
           4       0.81      0.83      0.82      1112
           5       0.90      0.82      0.85      1067
           6       0.82      0.91      0.86      1015
           7       0.87      0.89      0.88      1074
           8       0.83      0.74      0.78      1024
           9       0.86      0.77      0.81      1029

    accuracy                           0.86     10778
   macro avg       0.86      0.85      0.85     10778
weighted avg       0.86      0.86      0.85     10778
```

Results on [Scikit-learn digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) test partition
```
Classification report for classifier SVC(gamma=0.001):
              precision    recall  f1-score   support

           0       0.92      0.98      0.95        88
           1       0.98      0.44      0.61        91
           2       0.92      1.00      0.96        86
           3       0.84      0.84      0.84        91
           4       0.79      0.75      0.77        92
           5       0.94      0.51      0.66        91
           6       0.94      0.90      0.92        91
           7       0.98      0.45      0.62        89
           8       0.61      0.80      0.69        88
           9       0.41      0.91      0.57        92

    accuracy                           0.76       899
   macro avg       0.83      0.76      0.76       899
weighted avg       0.83      0.76      0.76       899
```

- Recognizing Written Alphabet Using a [Dataset](https://www.kaggle.com/datasets/sankalpsrivastava26/capital-alphabets-28x28/data) from Kaggle
[recognizing_written_alphabet.ipynb](toy_examples/comp_vision/recon_written_alphabet/recognizing_written_alphabet.ipynb) is trained using the [Alphabets Dataset (300x300)](https://www.kaggle.com/datasets/sankalpsrivastava26/capital-alphabets-28x28/data) dataset from Kaggle. For each character category, 1600 images are used for training and testing.Then, it is tested (validated) using its next half. Here is its performance:

```
Classification report for Kaggle dataset:
              precision    recall  f1-score   support

           A       0.88      0.90      0.89       808
           B       0.88      0.83      0.85       810
           C       0.93      0.89      0.91       797
           D       0.90      0.85      0.88       774
           E       0.81      0.82      0.81       789
           F       0.91      0.89      0.90       812
           G       0.58      0.87      0.70       794
           H       0.89      0.83      0.86       811
           I       0.76      0.91      0.83       768
           J       0.91      0.89      0.90       817
           K       0.91      0.86      0.88       811
           L       0.96      0.93      0.95       808
           M       0.96      0.93      0.94       813
           N       0.92      0.90      0.91       779
           O       0.91      0.95      0.93       797
           P       0.90      0.92      0.91       820
           Q       0.93      0.84      0.88       797
           R       0.93      0.85      0.89       832
           S       0.96      0.94      0.95       817
           T       0.96      0.93      0.95       841
           U       0.93      0.92      0.92       787
           V       0.92      0.90      0.91       798
           W       0.94      0.92      0.93       780
           X       0.96      0.88      0.92       791
           Y       0.92      0.89      0.90       790
           Z       0.92      0.91      0.92       772

    accuracy                           0.89     20813
   macro avg       0.90      0.89      0.89     20813
weighted avg       0.90      0.89      0.89     20813
```

## Articles Read
- [Easiest way to download kaggle data in Google Colab](https://www.kaggle.com/discussions/general/74235)
- [Recognizing hand-written digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)
- [Working With Text Data](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html)
- [Personality Detection using XLM-ROBERTa and Whisper](papers/Personality_Detection_Using_Xlm-Roberta_and_Whisper.pdf)  
- [Google ML Concepts](https://developers.google.com/machine-learning/crash-course/)
