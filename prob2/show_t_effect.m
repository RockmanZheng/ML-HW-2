fprintf('Show the effect of t in radial basis kernel on prediction surface.\n');
% Regularization parameter (Optimized)
lambda = 0.1950;
t=[1 10 100 505 1000 10000];
% Load in data for problem 1-a
load_data
[n,d]=size(X);
cut_idx = ceil(n/2);
train_idx = 1:cut_idx;
test_idx = cut_idx+1:n;

exp_times = length(t);
% Errors
train_error = zeros(1,exp_times);
test_error = zeros(1,exp_times);

for i=1:exp_times
    this_rbf=@(X,Y)(rbf_kernel(X,Y,t(i)));
    % Kernel regression
    alpha = kernel_regress(X(train_idx,:),y_noisy(train_idx),this_rbf,lambda);
    % Get train error
    y_train = kernel_pred(X(train_idx,:),X(train_idx,:),alpha,this_rbf);
    train_error(i)=norm(y_train-y_noisy(train_idx))/length(train_idx);
    % Get test error
    y_test = kernel_pred(X(test_idx,:),X(train_idx,:),alpha,this_rbf);
    test_error(i) = norm(y_test-y_noisy(test_idx))/length(test_idx);
    % Plot prediction surface and y_noisy(test_idx)
    ax = plot_pred_surf([min(X(:,1)),max(X(:,1));min(X(:,2)),max(X(:,2))],[100 100],X(train_idx,:),alpha,this_rbf);
    ax.ZLim=[min(y_noisy),max(y_noisy)];
    scatter3(ax,X(:,1),X(:,2),y_noisy);
    title(ax,sprintf('Prediction Surface & Data With t = %.1f',t(i)));
end

figure
p=plot(t,train_error,'LineWidth',2,'Marker','x');
ax = p.Parent;
hold on
plot(t,test_error,'LineWidth',2,'Marker','o');
% plot(lambda,true_error,'LineWidth',2,'Marker','*');
legend('Train','Test');
title('Mean Squared Error (MSE)');
xlabel('t');
ylabel('MSE');
% Change x scale into log to better visualize the trend of error with
% respect to t
ax.XScale = 'log';

fprintf('Conclusion:\n');
fprintf('In this file we will discuss the effect of t in radial basis kernel.\n'); 
fprintf('As shown in the plots, with smaller t, the prediction surface will be more peaky on a scatter of location, tending to over-fit the training data;\n');
fprintf('while with larger t, the prediction surface will tend to be flattened out, more smoothy. In fact, as t goes to +inf, radial basis kernel function will be constantly 1, and the resulting Gram matrix will be ones(n,n). And as can be proved, the prediction surface will then become a flat space (plane in 2d situation).\n');
fprintf('Thus in this case, the model tend to under-fit the data.\n');
fprintf('Therefore, an appropriate t should be chosen.\n');
