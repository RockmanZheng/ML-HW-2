fprintf('Exercise 1 (c): Test active learning method.\n');
% Load in data for problem 1-a
load_a1_data
[n,d]=size(X_in);

k1=5;
k2=10;

fprintf('Using no feature mapping\n');
% First feature mapping
X_map=feature_mapping(X_in,1);
X_train = [ones(n,1) X_map];
% Use active learning to choose data points
idx=active_learn(X_train,k1,k2);
theta  = linear_regress(y_noisy(idx),X_train(idx,:));
y_hat = linear_pred(theta,X_train);
error = norm(y_hat-y_true)/n;
fprintf('Mean Squared Error = %.4f\n',error);

fprintf('Using 2*log() feature mapping\n');
% Second feature mapping
X_map=feature_mapping(X_in,2);
X_train = [ones(n,1) X_map];
% Use active learning to choose data sets
idx=active_learn(X_train,k1,k2);
theta  = linear_regress(y_noisy(idx),X_train(idx,:));
y_hat = linear_pred(theta,X_train);
error = norm(y_hat-y_true)/n;
fprintf('Mean Squared Error = %.4f\n',error);
