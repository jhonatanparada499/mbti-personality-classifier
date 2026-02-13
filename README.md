# SP26_Jhonathan_parada
ML research with Jhonathan Parada

## Recognizing Written Digits Using a [Dataset](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download) from Kaggle

[recognizing_written_digits.ipynb](recon_written_digits/recognizing_written_digits.ipynb) is trained using half the [kaggle_written_digits](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download) dataset from Kaggle. Then, it is tested (validated) using its next half. Furthermore, to prove compatibility with other datasets, the [Scikit-learn digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) dataset from Scikit-learn is passed to the model to make predictions. The performance metrics for both cases are:

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

## Next Tasks

1. Combine Kaggle and Scikit-learn digits datasets to train and test model. (In progress...)
2. Train a Model to predict characters from the Alphabet. (Not started yet)
