function cvError = fuserLassoKfold(X, Y, lambda, tau, adjGroups, K)
%Alexey Ryabov, 2023
%k-fold validation of the fusedLasso regression
%the function returns cross validated error for given data set X and Y,
%regularization constant lambda and the weight tau for groups defined in adjGroups
% and K  folds partition of data into training and validation data
%you need this function to choose the best hyperparamter tau and lambda,
%for which the validation cost is the smallest
n = size(X, 1);
cv = cvpartition(n, 'KFold', K);
valError = NaN(K, 1);

for iK = 1:K
    %train
    indTrain = cv.training(iK);
    indTest = cv.test(iK);
    [Betas, trainError, valError(iK)] = fuserLassoSimple(X(indTrain, :), Y(indTrain, :), ...
        lambda, tau, adjGroups, X(indTest, :), Y(indTest, :));
end
%cross validated error
cvError = mean(valError);
