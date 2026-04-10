**Contact Information**  
Jhonatan Parada Torres  
jhonatanparada499@gmail.com  
  
**Category of Funded Work**  
NYSG Internship  
  
**Location**  
The City University of New York, Queensborough Community College  
11364  
11225  
  
**Description**  
Personality is a fundamental yet often overlooked aspect of human behavior. It influences daily decision making, interpersonal interactions, and overall compatibility between individuals. Among the most widely used frameworks for personality classification are the Myers–Briggs Type Indicator (MBTI) and the Big Five personality traits (OCEAN) model. These frameworks have broad applications across multiple domains, including therapy, education, law enforcement, customer service, human resources, and talent acquisition. Despite their widespread use, accurately identifying an individual’s personality type through traditional self-report assessments remains challenging due to subjectivity, bias, and limited reliability. To address these limitations, this project aims to develop a machine learning based approach for personality recognition by analyzing linguistic and speech patterns within textual data. Specifically, the project focuses on building an end-to-end system for personality classification based on the MBTI framework. The system will involve data collection, preprocessing, feature extraction, model training, and evaluation, ultimately enabling automated prediction of personality types from user-generated text. By leveraging advances in natural language processing and machine learning, this work seeks to provide a more scalable and data driven solution for personality assessment.  

**Progress, Challenges, and Next Steps**  
- sanitized and combined three datasets to train classification models.
- deployed end-to-end web application using 2 docker containers orchestrated with docker compose.
- created 4 binary classification models for each dimension in the MBTI framework.
- applied lemmatization on text to increase training speed to a total of 20 minutes using Google Colab.
- utilized 3 classification algorithms to train on the same dataset: Naïve Bayes Classifier, Support Vector Machine, and Logistic Regression.

**Accuracy by Classification Algorithms**
| Dimension | Naïve Bayes | SVM | Logistic Regression + tunned TFIDF params |
| -------- | ------- | ------- | ------- |
| IE | 0.75 | 0.75 | 0.88 | 
| NS | 0.90 | 0.90 | 0.92 | 
| TF | 0.64 | 0.86 | 0.91 | 
| JP | 0.58 | 0.74 | 0.85 | 
| Avg | 0.71 | 0.81 | 0.89 |

**Challenges**  
One of the challenges faced was dealing with an unbalanced dataset. When a model is trained using an unbalanced dataset, it might appear that it performs well when running predictions on a test set. But because the dataset is not evenly balanced (for example there is 90% records for class A, but only 10% for class B), the model will make correct predictions, not because of its algorithm, but because the dataset it was trained on is favoring one side.

To deal with that challenge, this is what I did: I added class_weight='balanced' in Logistic Regression parameter and added stratify=y in train_test_split. The first one, according to scikit-learn is a parameter used to handle unbalanced datasets, which adjusts its loss function to penalize mistakes on the minority class more than on the majority. The second one is a parameter that makes the train and test sets keep the same proportion of classes as the original dataset (if train set has ratio 9:1 classes, test set will also have 9:1 ratio, to prevent randomization function to accidently make the test ratio 10:0) 

**Feedback to NYSG**  

**Reflection**  

**Artifacts**  
