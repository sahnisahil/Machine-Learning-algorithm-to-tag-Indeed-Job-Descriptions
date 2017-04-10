m# Machine-Learning-algorith-to-tag-Indeed-Job-Descriptions
The project creates a machine learning algorithm to tag the job descriptions on indeed.

The data is first cleaned and different columns created for each tag to be predicted.

Then using the tm library in r, the job descriptions are cleaned for text mining to create variables and mine the important words from the raw data.

A Term Document matrix is created and the output is used to build the model.

After trying various predictive modeling methodologies, such as naive bayes', decision trees, random forest, J48, Decision stump, K-nearest neighbors and logistic regression with the ensemble of bagging and boosting , the best fit is found with the model using Weka adaboost with the J48 tree.
