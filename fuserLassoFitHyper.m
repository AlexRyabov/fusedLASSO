function [Betas, bestCVError, bestLambda, bestGamma, cvErrors] = ...
    fuserLassoFitHyper(X, Y, lambdas, gammas, adjGroups, Kfolds)
%% Alexey Ryabov, 2023
%% fused LASSO regression with optimization of the hyperparamters.
arguments
    X  %predictors
    Y  %target variable
    lambdas  %regularization constants
    gammas  %similarity weights
    adjGroups  %groups of regression coefficients
    Kfolds  %number of K folds for cross validation
end

%This is the main function for fused LASSO regression, if you do not know
%the exact values of hyperparameters lambda and gamma.
%use this function to test different combinations of lamdas and gammas to
%Using k-fold validation, this function finds the best set of the hyperparamters
% bestLambda, bestGamma and fits the fuged LASSO regression to the entire
% data set
%output
%Betas regresison coefficients, Ypredicted = Betas(1) + X * Betas(2:end)';
%bestCVError the minimal cross validation error obtained for the most optimal
%hypeparamters returned as
%bestLambda and bestGamma
%cvErrors the matrix of cross validation errors obtained for all
%combinations of lambdas (rows) and gammas (columns) of this matrix

cvErrors = NaN(length(lambdas), length(gammas));

for iG = 1:length(gammas)
    parfor iL = 1:length(lambdas)
        cvErrors(iL, iG) = fuserLassoKfold(X, Y, lambdas(iL), gammas(iG), adjGroups, Kfolds);
    end
end

% if length(lambdas) > 3 && length(gammas) > 3  %try lin regression
%     x2Lambda = log10(lambdas);
%     x1Gammas = log10(gammas);
%     [X1Gammas, X2Lambda] = meshgrid(x1Gammas, x2Lambda);
%     lm = fitlm([X1Gammas(:), X2Lambda(:)], cvErrors(:), 'quadratic');
%     f = @(x) lm.predict(x);
%     [bestParams] = fminunc(f, [x1Gammas(1), x2Lambda(1)]);
%     bestGamma = 10^bestParams(1);
%     bestLambda = 10^bestParams(2);
%     bestCVError = f([bestGamma, bestLambda]);
%     if bestGamma < gammas(1) || bestGamma > gammas(end) || bestLambda < lambdas(1) || bestLambda > lambdas(end)
%         warning('optimal params outside of the search range');
%         [bestCVError, indMinCVError] = min(cvErrors(:));
%         [indL, indG] = ind2sub(size(cvErrors), indMinCVError);
%         bestLambda = lambdas(indL);
%         bestGamma = gammas(indG);
%     end
% else
    [bestCVError, indMinCVError] = min(cvErrors(:));
    [indL, indG] = ind2sub(size(cvErrors), indMinCVError);
    bestGamma = gammas(indG);
    %find the lasgest lambda where cverror <= bestCVError*1.03
    indL = cvErrors(:, indG) <= bestCVError*1.003;
    bestLambda = max(lambdas(indL));
% end

[Betas, trainError] = fuserLassoSimple(X, Y, bestLambda, bestGamma, adjGroups);
