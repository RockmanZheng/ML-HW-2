fprintf('Exercise 1 (d): Compare 2 data query schemes: active learning and randomly selecting\n');
% Load in data for problem 1-a
load_a1_data
[n,d]=size(X_in);

k1=5;
k2=300;
exp_times = 50;
% MSE for active learning
error_1 = zeros(1,exp_times);
% MSE for random select
error_2 = zeros(1,exp_times);

fprintf('Using no feature mapping\n');
% First feature mapping
X_map=feature_mapping(X_in,1);
X_train = [ones(n,1) X_map];

for i=1:exp_times
    % Use active learning to choose data points
    idx=active_learn(X_train,k1,k2);
    theta  = linear_regress(y_noisy(idx),X_train(idx,:));
    y_hat = linear_pred(theta,X_train);
    error_1(i) = norm(y_hat-y_true)/n;
    % Randomly choose data points
    % Make sure the first k1 data points are the same
    temp=idx(1:k1);
    idx=randomly_select(X_train,k1,k2);
    idx(1:k1)=temp;
    theta  = linear_regress(y_noisy(idx),X_train(idx,:));
    y_hat = linear_pred(theta,X_train);
    error_2(i) = norm(y_hat-y_true)/n;
end
figure
plot(1:exp_times,error_1,'Color','b','LineWidth',2,'Marker','o');
hold on
plot(1:exp_times,error_2,'Color','r','LineWidth',2,'Marker','x');
plot(1:exp_times,mean(error_1)*ones(1,exp_times),'Color','b');
plot(1:exp_times,mean(error_2)*ones(1,exp_times),'Color','r');
xlabel('Experiment Times');
ylabel('MSE');
legend('Active Learn','Ramdom Select');
title('No Feature Mapping');

fprintf('Using 2*log() feature mapping\n');
% Second feature mapping
X_map=feature_mapping(X_in,2);
X_train = [ones(n,1) X_map];
for i=1:exp_times
    % Use active learning to choose data points
    idx=active_learn(X_train,k1,k2);
    theta  = linear_regress(y_noisy(idx),X_train(idx,:));
    y_hat = linear_pred(theta,X_train);
    error_1(i) = norm(y_hat-y_true)/n;
    % Randomly choose data points
    % Make sure the first k1 data points are the same
    temp=idx(1:k1);
    idx=randomly_select(X_train,k1,k2);
    idx(1:k1)=temp;
    theta  = linear_regress(y_noisy(idx),X_train(idx,:));
    y_hat = linear_pred(theta,X_train);
    error_2(i) = norm(y_hat-y_true)/n;
end
figure
plot(1:exp_times,error_1,'Color','b','LineWidth',2,'Marker','o');
hold on
plot(1:exp_times,error_2,'Color','r','LineWidth',2,'Marker','x');
plot(1:exp_times,mean(error_1)*ones(1,exp_times),'Color','b');
plot(1:exp_times,mean(error_2)*ones(1,exp_times),'Color','r');
xlabel('Experiment Times');
ylabel('MSE');
legend('Active Learn','Ramdom Select');
title('2*log() Feature Mapping');

fprintf('Conclusion: \n');
% fprintf('From Figure "No Feature Mapping", we can discover that\n');
% fprintf('In the case of no feature mapping, on average, MSE is lower when using active learning strategy;\n');
% fprintf('and variance in MSE is also smaller than that of using randomly selecting scheme.\n');
% fprintf('From Figure "2*log() Feature Mapping", we can discover that\n');
% fprintf('In the case of 2*log() feature mapping, on average, MSE is lower when using randomly selecting strategy;\n');
% fprintf('but variance in MSE is smaller when using active learning strategy.\n');
% fprintf('In summary, the variance in MSE will be reduced if we choose active learning strategy.\n');
% fprintf('But this may introduce unpredictable bias in MSE.\n');
% fprintf('The sign and magnitude of this bias is affected by data set itself, and feature mapping we are using.\n');
fprintf('From the plotted figures, we can discovered that with both feature\n');
fprintf('mappings,the average and variance of prediction MSE are lower when\n');
fprintf('using active learning strategy. That is to say, with active\n');
fprintf('learning strategy helping us select data, we can generally obtain \n');
fprintf('more precise model.\n');
fprintf('In this experiment, the first k1 = %d data are collected randomly;\n',k1);
fprintf('then the following k2 = %d data are collected according to differ-\n',k2);
fprintf('-ent schemes. The first scheme is active learning, and the other\n');
fprintf('just ramdomly selecting.\n');
fprintf('Note that the first k1 data are the same across 2 schemes.\n');
fprintf('But remember, the effect of active learning stratege - that is \n');
fprintf('reducing MSE in parameter estimates, or in other words, improving\n');
fprintf('the precision of prediction - only occurs when a large portion of\n');
fprintf('data is collected under its guidence (i.e., ratio k1/k2 must be \n');
fprintf('small). Otherwise, the effect of randomly selecting will just take\n');
fprintf('over\n');