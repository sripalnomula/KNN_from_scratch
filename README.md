# KNN_from_scratch
## K-Nearest Neighbors Classification(KNN)
For this part we are given with a task to implement KNN from scratch without using any inbuilt libraries except numpy,test our code and them compare the results with scikit-learn implmentation.

### KNN classification: 
The K-Nearest Neighbors algorithm is a supervised machine learning algorithm which is used to solve both classification and regression problems.

### steps:
* 1.As we are already provided with skeleton code in two files utils.py and K-Nearest_neighbors.py so in here i have implemented funtions 
* 2.for euclidean_distance between 2 vectors: sum((x1-x2)^2)
  for manhattan_dist between 2 vectors: sum(abs(x1-x2))
* 3.In the main K_nearest_neighbors.py we already have a class which has attributes n_neighbors(to represent the number of neighbors (i.e.,5 or 10 etc.,) to compare it with sample while predicting class values.
* 4.weight function which can be either "uniform" or "distance" and when the parameter is distance weights assigned will be proportional to the inverse of the distance.
* 5.metric is either set to be 11 or 12, where 11 is passed as attribute it considers Manhattan distance else Euclidean distance.
other attributes are as:
 _X: A numpy array of shape (n_samples,n_features) representing the input data used when fitting the model and predicting target class values.

 _y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data used when fitting the model and predicting target class values.

 _distance: An attribute representing which distance metric is used to calculate distances between samples. This is set when creating the object to either the euclidean_distance or manhattan_distance functions defined in utils.py based on what argument is passed into the metric parameter of the class.

* 6.we then implemented the fit function to fit the model to the provided data matrix and targets(As there is nothing to train as KNN is instance based instead of model so just assigned then to class attributes)
* 7.next moved to predict function: 
Intialize a empty list out_labels to store repective argmax values and return the list when called upon

### Testing:
when tested the code gave the similiar output as scikit-learn implementation.
