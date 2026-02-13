# SP26_Jhonathan_parada
ML research with Jhonathan Parada

## Recognizing Written Digits Using a [Dataset](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download) from Kaggle

[recognizing_written_digits.ipynb](recon_written_digits/recognizing_written_digits.ipynb) is trained using half the dataset [kaggle_written_digits](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download) from Kaggle. Then, it is tested (validated) using the next half of the dataset. Furthermore, to prove compatibility with other datasets, the [Scikit-learn digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) dataset from Scikit-learn is passed to the model to make predictions. The performance metrics for both cases are:

Results on [kaggle_written_digits](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9?resource=download) test partiton
```
Classification report for classifier SVC(gamma=0.01):
              precision    recall  f1-score   support

           0       0.90      0.90      0.90      1112
           1       0.80      0.91      0.85      1109
           2       0.86      0.82      0.84      1130
           3       0.76      0.78      0.77      1094
           4       0.75      0.81      0.78      1079
           5       0.84      0.72      0.77      1104
           6       0.77      0.89      0.83      1081
           7       0.84      0.85      0.84      1051
           8       0.82      0.73      0.77      1045
           9       0.80      0.70      0.75       973

    accuracy                           0.81     10778
   macro avg       0.81      0.81      0.81     10778
weighted avg       0.82      0.81      0.81     10778
```

Results on [Scikit-learn digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) test partition
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
