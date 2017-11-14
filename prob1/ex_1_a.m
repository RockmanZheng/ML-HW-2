fprintf('Exercise 1 (a): Test linear regression.\n');
% Load in data for problem 1-a
load_a1_data
[n,d]=size(X_in);
X_train = [ones(n,1) X_in];
fprintf('Running linear regression...\n');
theta  = linear_regress(y_noisy,X_train);
fprintf('Predicting...\n');
y_hat = linear_pred(theta,X_train);
error = norm(y_hat-y_true)/n;
fprintf('Mean Squared Error = %.4f\n',error);

%clf
%figure
%[X_sort,Idx]=sort(X_in(:,1));
%plot(X_sort,y_hat(Idx));
%plot(1:n,y_hat);
%hold on
%scatter(X_sort,y_noisy(Idx));
%plot(1:n,y_true);
%scatter(1:n,y_noisy);

%printf('Error = %f',error);