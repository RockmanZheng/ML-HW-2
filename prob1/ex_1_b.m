fprintf('Exercise 1 (b): Compare 2 feature mapping.\n');

% Load in data for problem 1-a
fprintf('Loading data...\n');
load_a1_data
[n,d]=size(X_in);

fprintf('Using no feature mapping\n');
% First feature mapping
X_map=feature_mapping(X_in,1);
X_train = [ones(n,1) X_map];
theta  = linear_regress(y_noisy,X_train);
y_hat = linear_pred(theta,X_train);
error = norm(y_hat-y_true)/n;
fprintf('Mean Squared Error = %.4f\n',error);

fprintf('Using 2*log() feature mapping\n');
% Second feature mapping
X_map=feature_mapping(X_in,2);
X_train = [ones(n,1) X_map];
theta  = linear_regress(y_noisy,X_train);
y_hat = linear_pred(theta,X_train);
error = norm(y_hat-y_true)/n;
fprintf('Mean Squared Error = %.4f\n',error);