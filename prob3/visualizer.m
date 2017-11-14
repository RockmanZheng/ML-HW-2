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

%% Radial Basis Kernel
t=0.03;
fprintf('Choosing t = %.2f in radial basis kernel function\n',t);
this_kernel = @(X,Y)(rbf_kernel(X,Y,t));
[alpha_b,error_count_b] = train_kernel_perceptron(X_train_b,y_train_b,this_kernel);
train_error_b = error_count_b/length(y_train_b);
f_b = discriminant_function(alpha_b,X_train_b,this_kernel,X_test_b);
test_error_b = length(find(sign(f_b)-y_test_b))/length(f_b);
ax = plot_dec_boundary([min(X_b(:,1)) max(X_b(:,1));min(X_b(:,2)) max(X_b(:,2))],[100 100],X_train_b,alpha_b,this_kernel);
plot_dec_surf([min(X_b(:,1)) max(X_b(:,1));min(X_b(:,2)) max(X_b(:,2))],[100 100],X_train_b,alpha_b,this_kernel);
neg_idx = find(y_b<0);
pos_idx = find(y_b>0);
scatter(ax,X_b(neg_idx,1),X_b(neg_idx,2),20,'r');
scatter(ax,X_b(pos_idx,1),X_b(pos_idx,2),20,'b');

t=0.1;
fprintf('Choosing t = %.2f in radial basis kernel function\n',t);
this_kernel = @(X,Y)(rbf_kernel(X,Y,t));
[alpha_c,error_count_c] = train_kernel_perceptron(X_train_c,y_train_c,this_kernel);
train_error_c = error_count_c/length(y_train_c);
f_c = discriminant_function(alpha_c,X_train_c,this_kernel,X_test_c);
test_error_c = length(find(sign(f_c)-y_test_c))/length(f_c);
ax = plot_dec_boundary([min(X_c(:,1)) max(X_c(:,1));min(X_c(:,2)) max(X_c(:,2))],[100 100],X_train_c,alpha_c,this_kernel);
plot_dec_surf([min(X_c(:,1)) max(X_c(:,1));min(X_c(:,2)) max(X_c(:,2))],[100 100],X_train_c,alpha_c,this_kernel);
neg_idx = find(y_c<0);
pos_idx = find(y_c>0);
scatter(ax,X_c(neg_idx,1),X_c(neg_idx,2),20,'r');
scatter(ax,X_c(pos_idx,1),X_c(pos_idx,2),20,'b');

fprintf('Training error on banana data: %.2f%%, training error on circle data: %.2f%%\n',train_error_b*100,train_error_c*100);
fprintf('Testing error on banana data: %.2f%%, testing error on circle data: %.2f%%\n',test_error_b*100,test_error_c*100);

%% Polynomial Basis Kernel
% Banana data set
d=16;
fprintf('Choosing d = %d in polynomial kernel function\n',d);
this_kernel = @(X,Y)(poly_kernel(X,Y,d));
[alpha_b,error_count_b] = train_kernel_perceptron(X_train_b,y_train_b,this_kernel);
train_error_b = error_count_b/length(y_train_b);
f_b = discriminant_function(alpha_b,X_train_b,this_kernel,X_test_b);
test_error_b = length(find(sign(f_b)-y_test_b))/length(f_b);
ax = plot_dec_boundary([min(X_b(:,1)) max(X_b(:,1));min(X_b(:,2)) max(X_b(:,2))],[100 100],X_train_b,alpha_b,this_kernel);
% plot_dec_surf([min(X_b(:,1)) max(X_b(:,1));min(X_b(:,2)) max(X_b(:,2))],[100 100],X_train_b,alpha_b,this_kernel);
neg_idx = find(y_b<0);
pos_idx = find(y_b>0);
scatter(ax,X_b(neg_idx,1),X_b(neg_idx,2),20,'r');
scatter(ax,X_b(pos_idx,1),X_b(pos_idx,2),20,'b');

% Circle data set
d=14;
fprintf('Choosing d = %d in polynomial kernel function\n',d);
this_kernel = @(X,Y)(poly_kernel(X,Y,d));
[alpha_c,error_count_c] = train_kernel_perceptron(X_train_c,y_train_c,this_kernel);
train_error_c = error_count_c/length(y_train_c);
f_c = discriminant_function(alpha_c,X_train_c,this_kernel,X_test_c);
test_error_c = length(find(sign(f_c)-y_test_c))/length(f_c);
ax = plot_dec_boundary([min(X_c(:,1)) max(X_c(:,1));min(X_c(:,2)) max(X_c(:,2))],[100 100],X_train_c,alpha_c,this_kernel);
% plot_dec_surf([min(X_c(:,1)) max(X_c(:,1));min(X_c(:,2)) max(X_c(:,2))],[100 100],X_train_c,alpha_c,this_kernel);
neg_idx = find(y_c<0);
pos_idx = find(y_c>0);
scatter(ax,X_c(neg_idx,1),X_c(neg_idx,2),20,'r');
scatter(ax,X_c(pos_idx,1),X_c(pos_idx,2),20,'b');

fprintf('Training error on banana data: %.2f%%, training error on circle data: %.2f%%\n',train_error_b*100,train_error_c*100);
fprintf('Testing error on banana data: %.2f%%, testing error on circle data: %.2f%%\n',test_error_b*100,test_error_c*100);



