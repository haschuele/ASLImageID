# ASLImageID

There is a need for real-time sign language translation as interpreters are few and expensive. This is especially a problem in education where deaf or hard-of-hearing students and staff require interpretation services. Our team explored a solution by classifying the American Sign Language (ASL) alphabet using 65,774 RGB images taken from 5 different individuals. 

We used multiclass logistic regression containing a sequential layer and a softmax activation as our baseline. We used 20 epochs to get a final validation accuracy of 0.555. We also tested a binary logistic regression model, which had much better accuracy when focused on a single letter. However, we wanted a model that could accurately predict all 24 letters.

This led us to CNN models, where we started by adding different Conv2D, max pooling, and dense layers. Of the models whose training and validation accuracies converged, we selected our best performing layer combination and experimented with different hyperparameters - including kernel size, pool size, strides, learning rate, and optimizer. Some modifications (i.e. learning rate = 0.01 and optimizer = SGD) did very poorly and we didn’t bother incorporating them for subsequent tests. Our best CNN model had 94% accuracy on the test data.
