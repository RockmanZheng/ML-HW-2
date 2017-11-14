fprintf('Exercise (3): Test kernel perceptron algorithm\n');
% Load in data
load_data
% Split data into training set and test set
% Get class index
neg_idx = find(y_b<0)';    % Negative class
pos_idx = find(y_b>0)';    % Positive class
neg_cut = ceil(length(neg_idx)/2);  % Cut from middle
pos_cut = ceil(length(pos_idx)/2);
X_train_b = X_b([neg_idx(1:neg_cut) pos_idx(1:pos_cut)],:);
X_test_b = X_b([neg_idx(neg_cut+1:end) pos_idx(pos_cut+1:end)],:);
y_train_b = y_b([neg_idx(1:neg_cut) pos_idx(1:pos_cut)]);
y_test_b = y_b([neg_idx(neg_cut+1:end) pos_idx(pos_cut+1:end)]);
% Get class index
neg_idx = find(y_c<0)';    % Negative class
pos_idx = find(y_c>0)';    % Positive class
neg_cut = ceil(length(neg_idx)/2);  % Cut from middle
pos_cut = ceil(length(pos_idx)/2);
X_train_c = X_c([neg_idx(1:neg_cut) pos_idx(1:pos_cut)],:);
X_test_c = X_c([neg_idx(neg_cut+1:end) pos_idx(pos_cut+1:end)],:);
y_train_c = y_c([neg_idx(1:neg_cut) pos_idx(1:pos_cut)]);
y_test_c = y_c([neg_idx(neg_cut+1:end) pos_idx(pos_cut+1:end)]);


t=[0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100];
exp_times = length(t);
train_error_b = zeros(1,exp_times);
train_error_c = zeros(1,exp_times);
test_error_b = zeros(1,exp_times);
test_error_c = zeros(1,exp_times);
for i=1:exp_times
    fprintf('Choosing t = %.2f in radial basis kernel function\n',t(i));
    this_kernel = @(X,Y)(rbf_kernel(X,Y,t(i)));
    % Train perceptron
    [alpha_b,error_count_b] = train_kernel_perceptron(X_train_b,y_train_b,this_kernel);
    [alpha_c,error_count_c] = train_kernel_perceptron(X_train_c,y_train_c,this_kernel);
    % Compute training error
    train_error_b(i) = error_count_b/length(y_train_b);
    train_error_c(i) = error_count_c/length(y_train_c);
    fprintf('Training error on banana data: %.2f%%, training error on circle data: %.2f%%\n',train_error_b(i)*100,train_error_c(i)*100);
    % Compute test error
    f_b = discriminant_function(alpha_b,X_train_b,this_kernel,X_test_b);
    f_c = discriminant_function(alpha_c,X_train_c,this_kernel,X_test_c);
    test_error_b(i) = length(find(sign(f_b)-y_test_b))/length(f_b);
    test_error_c(i) = length(find(sign(f_c)-y_test_c))/length(f_c);
    fprintf('Testing error on banana data: %.2f%%, testing error on circle data: %.2f%%\n',test_error_b(i)*100,test_error_c(i)*100);
end

% Plot training and testing errors
figure
p=plot(t,train_error_b,'LineWidth',2,'Marker','x');
ax = p.Parent;
hold on
plot(t,test_error_b,'LineWidth',2,'Marker','o');
xlabel('t');
ylabel('Error');
ax.XScale='log';
legend('Train','Test');
title('Error Plot For Banana Data Set');
figure
p=plot(t,train_error_c,'LineWidth',2,'Marker','x');
ax = p.Parent;
hold on
plot(t,test_error_c,'LineWidth',2,'Marker','o');
xlabel('t');
ylabel('Error');
ax.XScale='log';
legend('Train','Test');
title('Error Plot For Circle Data Set');
% Plot decision boundary and data points to visualize perceptron model


% Here try using polynomial kernel
d = 1:30;
exp_times = length(d);
train_error_b = zeros(1,exp_times);
train_error_c = zeros(1,exp_times);
test_error_b = zeros(1,exp_times);
test_error_c = zeros(1,exp_times);
for i=1:exp_times
    fprintf('Choosing d = %d in polynomial kernel function\n',d(i));
    this_kernel = @(X,Y)(poly_kernel(X,Y,d(i)));
    % Train perceptron
    [alpha_b,error_count_b] = train_kernel_perceptron(X_train_b,y_train_b,this_kernel);
    [alpha_c,error_count_c] = train_kernel_perceptron(X_train_c,y_train_c,this_kernel);
    % Compute training error
    train_error_b(i) = error_count_b/length(y_train_b);
    train_error_c(i) = error_count_c/length(y_train_c);
    fprintf('Training error on banana data: %.2f%%, training error on circle data: %.2f%%\n',train_error_b(i)*100,train_error_c(i)*100);
    % Compute test error
    f_b = discriminant_function(alpha_b,X_train_b,this_kernel,X_test_b);
    f_c = discriminant_function(alpha_c,X_train_c,this_kernel,X_test_c);
    test_error_b(i) = length(find(sign(f_b)-y_test_b))/length(f_b);
    test_error_c(i) = length(find(sign(f_c)-y_test_c))/length(f_c);
    fprintf('Testing error on banana data: %.2f%%, testing error on circle data: %.2f%%\n',test_error_b(i)*100,test_error_c(i)*100);
end

% Plot training and testing errors
figure
p=plot(d,train_error_b,'LineWidth',2,'Marker','x');
ax = p.Parent;
hold on
plot(d,test_error_b,'LineWidth',2,'Marker','o');
xlabel('t');
ylabel('Error');
% ax.XScale='log';
legend('Train','Test');
title('Error Plot For Banana Data Set');
figure
p=plot(d,train_error_c,'LineWidth',2,'Marker','x');
ax = p.Parent;
hold on
plot(d,test_error_c,'LineWidth',2,'Marker','o');
xlabel('d');
ylabel('Error');
% ax.XScale='log';
legend('Train','Test');
title('Error Plot For Circle Data Set');
% Plot decision boundary and data points to visualize perceptron model

