fprintf('Perform a series of experiments to optimize hyperparameters which yields the lowest test error.\n');
% Load in data for problem 1-a
load_data
[n,d]=size(X);
cut_idx = ceil(n/2);
train_idx = 1:cut_idx;
test_idx = cut_idx+1:n;


lambda = [0.03 0.1 0.15 0.175 0.2 0.225 0.25 0.3 0.4 0.5 0.8 1 3]';
t=[1 5 10 20 100 200 300 400 450 475 500 525 550 600 700 1000]';
m=length(lambda);
n=length(t);
test_error = zeros(m,n);



for j=1:n
    % Construct rbf kernel
    this_rbf=@(X,Y)(rbf_kernel(X,Y,t(j)));
    for i=1:m
        % Kernel regression
        alpha = kernel_regress(X(train_idx,:),y_noisy(train_idx),this_rbf,lambda(i));
        % Get test error
        y_test = kernel_pred(X(test_idx,:),X(train_idx,:),alpha,this_rbf);
        test_error(i,j) = norm(y_test-y_noisy(test_idx))/length(test_idx);
    end
end

[M,I1]=min(test_error);
[M,I2]=min(M);
optimal_lambda = lambda(I1(I2));
optimal_t=t(I2);
fprintf('Optimal t = %.2f, Optimal lambda = %.2f, Test Error = %.3f\n',optimal_t,optimal_lambda,M);

[Lambda,T]=meshgrid(lambda,t);
figure
surf(Lambda,T,test_error');
shading interp
xlabel('\lambda');
ylabel('t');
zlabel('Test MSE');
figure
contour(Lambda,T,test_error','ShowText','on');
xlabel('\lambda');
ylabel('t');
%zlabel('Test MSE');

% Refined

lambda = 0.1:1e-3:0.3;
t=500:1:510;
m=length(lambda);
n=length(t);
test_error = zeros(m,n);

for j=1:n
    % Construct rbf kernel
    this_rbf=@(X,Y)(rbf_kernel(X,Y,t(j)));
    for i=1:m
        % Kernel regression
        alpha = kernel_regress(X(train_idx,:),y_noisy(train_idx),this_rbf,lambda(i));
        % Get test error
        y_test = kernel_pred(X(test_idx,:),X(train_idx,:),alpha,this_rbf);
        test_error(i,j) = norm(y_test-y_noisy(test_idx))/length(test_idx);
    end
end

[M,I1]=min(test_error);
[M,I2]=min(M);
optimal_lambda = lambda(I1(I2));
optimal_t=t(I2);
fprintf('Optimal t = %.2f, Optimal lambda = %.4f, Test Error = %.5f\n',optimal_t,optimal_lambda,M);

[Lambda,T]=meshgrid(lambda,t);
figure
surf(Lambda,T,test_error');
shading interp
xlabel('\lambda');
ylabel('t');
zlabel('Test MSE');
figure
contour(Lambda,T,test_error','ShowText','on');
xlabel('\lambda');
ylabel('t');
%zlabel('Test MSE');
        
        