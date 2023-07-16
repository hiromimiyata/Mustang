# Mustang
Horse winning prediction model

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
