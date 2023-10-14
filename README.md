# fusedLASSO

This is a simple fused lasso regression to fit a linear combination of features

  Y^j = \sum_i \theta_i X_i^j
  
  where the cost function is 
  
  J = mean((Y- X*thetas').^2)/2 + lambda * sum(abs(thetas(2:end)))  + tau * sum(abs(diff(thetas(2:end))).^2.*groups);




Use this for performing a simple fused lasso regression and get training and test error.  The test data Xtest, Ytest are optional paramters 

    [Thetas, trainError, testError] = fuserLassoSimple(Xtrain, Ytrain, lambda, tau, adjGroups, Xtest, Ytest);

adjGroups should be a matrix with two columns including indexes of features to fuse 

You can define this manually, for example, if you want to fuse the following features together: 1&2, 2&3, 3&5 
you should define 

    adjGroups = [1, 2; 2, 3; 3, 5];

to faster define groups of fused features you can use fuserLassoAdjGroups 

    [adjGroupsij] = fuserLassoAdjGroups(groups);

Assume that we want to fuse from  X_1...X_5 and X_6...X_8
then we define groups = [1, 5; 6, 8]  and after calling 

    [adjGroupsij] = fuserLassoAdjGroups(groups);
 
 we get 
 
     adjGroupsij =
     1     2
     2     3
     3     4
     4     5
     6     7
     7     8

To tune hypeparamters you can use functions which find k-fold cross validation error.        

Use this for finding cross validation error using k-fold validation for given lambda and tau
 
     cvErrors(iL, iG) = fuserLassoKfold(X, Y, lambda, tau, adjGroups, Kfolds);

Use this for selecting the most optimal lambda and tau using k fold validation across values in vectors lambdas and taus

     [Betas, bestCVError, bestLambda, bestGamma, cvErrors] = fuserLassoFitHyper(X, Y, lambdas, taus, adjGroupsij, kFolds);

please send me an email if you need a help. alryabov@gmail.com


1.  F. Dondelinger, S. Mukherjee, The Alzheimer’s Disease Neuroimaging Initiative, The joint lasso: high-dimensional regression for group structured data. Biostatistics. 21, 219–235 (2020).

2.  R. Tibshirani, M. Saunders, S. Rosset, J. Zhu, K. Knight, Sparsity and Smoothness Via the Fused Lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology. 67, 91–108 (2005).


  
