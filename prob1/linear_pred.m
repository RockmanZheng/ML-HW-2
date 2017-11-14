function y_hat = linear_pred(theta,X_test)
% We are not explicitly including the offset parameter but instead rely on
% the feature vectors to provide a constant component

y_hat = X_test*theta;

end