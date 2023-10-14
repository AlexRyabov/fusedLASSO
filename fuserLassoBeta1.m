function [Betas, trainError, valError] = fuserLassoBeta1(X, Y, Beta1, lambda, tau, adjGroups, Xval, Yval)
%Alexey Ryabov, 2023
%same as fuserLassoBeta1, but does not change the Intercept, which should be defined 

arguments
    X  %predictors
    Y  %target variable
    Beta1 % intercept
    lambda  %regularization
    tau  %weights for
    adjGroups  %groups
    Xval = []; %validation data
    Yval = [];
end

%shift the index of adjGroups by one because of the bias term
if ~isempty(adjGroups)
    adjGroupsBias = adjGroups + 1;
end

n = size(X, 1);
X2fit =  [ones(n, 1), X];
m = size(X2fit, 2);
pow = 2;
%cost function
%J = @(thetas) mean((Y- X2fit*thetas').^2)/2 + lambda * sum(abs(thetas(2:end)))...
%   + tau * sum(abs(thetas(adjGroupsBias(:, 2)) - thetas(adjGroupsBias(:, 1)) ).^2 );
groups = zeros(1, m-2);
groups(adjGroups(:, 1)) = 1;

J = @(thetas) mean((Y- X2fit*thetas').^2)/2 + lambda * sum(abs(thetas(2:end)))...
    + tau * sum(abs(diff(thetas(2:end))).^pow.*groups);
JBeta1 = @(BetasSlopes) J([Beta1, BetasSlopes]);

%% minimize the cost function
% find the most optimal set of fused LASSO regression coefficients
% if the fminunc function does not give a good minimum, set
% SwarmMinimization=1  to use particle swarm minimization
 Minimizer = 'fmin';
% Minimizer = 'SuiteLasso';  %produces much more stable solutions 
%https://github.com/MatOpt/SuiteLasso
switch Minimizer
    case 'Swarm' %use swarm minimization
        options = optimoptions('particleswarm','SwarmSize',50,'HybridFcn',@fmincon);
        Betas = particleswarm(JBeta1,m, -10*ones(1, m-1), 10*ones(1, m-1), options);
    case 'fmin'  %use fminunc function
        %for me this simple minimization worked perfectly:)
        %options = optimoptions('fminunc','MaxFunctionEvaluations', 500*m);
        Betas = fminunc(JBeta1,0.2*rand(1, m-1)-0.1);
  
end
Betas = [Beta1, Betas];
%training error
trainError  = mean((Y- X2fit*Betas').^2)/2;

%validation error
if ~isempty(Xval)
    valError = mean(((Betas(1) + Xval*Betas(2:end)') - Yval).^2)/2;
end

