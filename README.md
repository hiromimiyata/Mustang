# Mustang


# Horse Racing prediction model ( father_res_place.ipynb)


This repo contains a machine learning model that predicts the placement of horses in a race based on father's features. The model achieved an accuracy of 96% on the test set.



# Model
 
The prediction model is based on logistic regression, a supervised learning algorithm commonly used for binary classification problems. Logistic regression models the probability of the positive class (horse placed) based on the input feature.


To train the model, the dataset was preprocessed by performing one-hot encoding on the 'father' column and concatenating the encoded columns with the original dataframe. The dataset was then split into training and testing sets using a test size of 20%.

The logistic regression model was trained on the training set and achieved an accuracy of 96% on the test set. The accuracy metric measures the percentage of correctly predicted placements. The model performed well in distinguishing between placed and non-placed horses.


Confusion Matrix:
[[20785   871]
 [  210  8037]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.99      0.96      0.97     21656
         1.0       0.90      0.97      0.94      8247

    accuracy                           0.96     29903
   macro avg       0.95      0.97      0.96     29903
weighted avg       0.97      0.96      0.96     29903





From the confusion matrix, we can see that the model accurately predicted 20,785 non-placed horses (true negatives) and 8,037 placed horses (true positives). It also made 871 false positive predictions (non-placed horses predicted as placed) and 210 false negative predictions (placed horses predicted as non-placed).




# Conclusion: 

The horse racing prediction model based on logistic regression achieved an accuracy of 96% on the test set. It demonstrated strong performance in classifying whether a horse will be placed or not in a race. The confusion matrix provides additional insights into the model's performance, showing the distribution of correct and incorrect predictions based on father's features.








### Horse winning prediction model (horse_racing ipynb)



This model aims to predict whether a horse will win or not based on the names of its father and mother. We believe that the genetic information inherited from the horse's parents might impact its performance and speed.

How it Works:
We collected data on horses, including the names of their father and mother, as well as whether they won or not.(res_win)
Using machine learning techniques, we built two models: one using a decision tree algorithm and another using a neural network with multiple hidden layers.
The decision tree model achieved an accuracy of approximately 88%, while the neural network model achieved an accuracy of around 90%.
We evaluated the models using classification reports, which provide information on precision, recall, and F1-score.
Based on the classification reports, the models performed reasonably well in predicting winning horses, with precision ranging from 12% to 19% for the neural network model and 18% for the decision tree model.
The models were trained on a dataset that included information about the father and mother of each horse, allowing them to learn patterns and make predictions based on this genetic information.

Conclusion: 

Using the names of the horse's father and mother as features in the prediction models was an interesting approach to explore the potential genetic influence on racing ability. While the models achieved decent accuracy, predicting horse race outcomes only based solely on the names of their parents can be unreliable and needs to look at different factors as well such as track conditions, and past performance etc. 
