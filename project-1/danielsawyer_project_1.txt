# Daniel Sawyer U3363-7705, Project-1

## Question 1:
```
Use J48 (decision tree learning algorithms) and load the Iris data (in the files).  Choose J48 as the classifier. Use default options.  For test options use the training set. What accuracy do you get?  Does analysis  allow for any conclusions about this classifier applied to this data set?
```
### Answer 1:
```
98%
You can tell from the confusion matrix that there is relatively high variance but possibly some bias between class b and c (Iris-versicolor and Iris-virginica)
```


## Question 2:
```
Now try a 10-fold cross validation.  What accuracy do you get? 
```
### Answer 2:
```
96%
```


## Question 3:
```
Now change the minnumobj to 1.  That controls the minimum number of examples at a leaf.  What do you get?   Why?
```
### Answer 3:
```
94.6667%
Reducing minNumObj to one allows minimum of 1 example at a leaf, which reduces the accuracy since you lose percision due to the selection rule being more specific.
```


## Question 4:
```
Now change minnumbobj back to 2.  Click on more options and change the seed to 23.  Do a 10-fold cross validation.  What accuracy do you get? . Did the seed matter in this accuracy?
```
### Answer 4:
```
95.3333%
Yes, the accuracy is slightly affected by the seed value since changing the random seed value changes the sampling
```


## Question 5:
```
Change the seed back to 1.
Now go to select attributes and apply Principal Components.  Do all 4 original features have coefficients that are distinct from 0?   Looking at the first  principal component what seem to be the most important features and why?
```
### Answer 5:
```
Yes, all are distinct from zero
The most important features are petal-length and petal-width. You can infer this from the their coefficients are the larges positive numbers.
```


## Question 6:
```
Now go to preprocess and delete the features having to do with petal.  What is your accuracy?
```
### Answer 6:
```
72.6667 %
```


## Question 7:
```
Reload and delete the features having to do with sepal.  What is your accuracy?  
```
### Answer 7:
```
96%
```


## Question 8:
```
Did the weights for Principal components agree with the above results (explain)?
```
### Answer 8:
```
Yes, since the PCA showed that the petal information was more valuable than sepal related feature. The accuracy of the model without sepal features shows the PCA was correct.
```


## Question 9:
```
Now let’s cluster the Iris data.  Load it (remember to remove the classes), choose EM clustering.  Cluster and visualize in petal-length, petal-width space.    How many clusters were found?  Can you conclude anything from the number?
```
### Answer 9:
```
5 clusters were found.
Well, since there are only 3 classes but it is cluster them into 5, there is a chance that there is some feature variance in the same class, so there is some confusion.
```


## Question 10:
```
Next use EM and set it to 3 clusters. Visualize it.  Visualize the raw data all in petal-length, petal-width space.  Can you see any errors and if so, describe them  (e.g. between what classes). 
```
### Answer 10:
```
Yes, there seems to be some over lap between Iris-versicolor and Iris-virginica, so there is low variance between them and some confusion. 
```


## Question 11:
```
Use simple K-means for clustering and set it to 3 clusters (remember remove classes) with the rest of the parameters as default.  How many errors in your result?  How does this compare with EM and 3 clusters?  You need to visualize the original Iris data with classes.
```
### Answer 11:
```
Sum of squared errors: 6.998114004826762
Compared to EM with 3 clusters, there isnt a big difference. There are a few from cluster 0 and cluster 2 that have switched between each other but that would be expected because the 2 classes, Iris-versicolor and Iris-virginica, are so close to each other.
```