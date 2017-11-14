function theta = linear_regress(y,X,lambda)
% Linear regression, result: X*theta approx. y
% y: responses
% X: input signal
% lambda: regularization parameter
% theta: estimated parameter
% We are not explicitly including the offset parameter but instead rely on
% the feature vectors to provide a constant component

if nargin == 2
    % Default value for regularization parameter
    % No regularization
    lambda = 0;
end

[~,d]=size(X);

theta = linsolve(lambda*eye(d)+X'*X,X'*y);

end