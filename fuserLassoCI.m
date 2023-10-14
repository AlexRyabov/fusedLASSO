function [BetasCI, BetasCIStat] = fuserLassoCI(X, Y, lambda, gamma, adjGroups, nboot)
%Alexey Ryabov, 2023
%get confidence intervals for fused LASSO regression of data in X, Y (in columns)
% bootstrap sequnces of X and Y are used to get the CI intervals of Betas
% 
% regularization
%constant lambda and gamma and groups defined by adjGroups
%the function returns median coefficients Betas for , so Ypredicted = Betas(1) + X*Betas(2:end)
%and BetasCI which are 0.05 and 0.95 quntiles of betas obtained for various bootstrap sequnces of the data 
% meanTrainError is the mean training error and meanR2 is the 

arguments
    X  %predictors
    Y  %target variable
    lambda  %regularization
    gamma  %weights ford

    adjGroups  %groups
    nboot = 100 % number of Number of bootstrap samples
end


fitLASSO = @(X) fuserLassoSimple(X(:, 1:end-1), X(:, end), lambda, gamma, adjGroups);
[BetasCI, BetasCIStat] = bootci(nboot,fitLASSO, [X, Y]);



