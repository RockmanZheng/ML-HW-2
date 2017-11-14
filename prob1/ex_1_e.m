fprintf('Exercise 1 (e): Discover active learning chooses points at the extrema of available space\n');
% Load in data for problem 1-a
load_a1_data
[n,d]=size(X_in);

k1=5;
k2=300;

fprintf('Using no feature mapping\n');
% First feature mapping
X_map=feature_mapping(X_in,1);
X_train = [ones(n,1) X_map];

% Use active learning to choose data points
idx=active_learn(X_train,k1,k2);
plot_points(X_map,idx,setdiff(1:n,idx));

fprintf('Using 2*log() feature mapping\n');
% Second feature mapping
X_map=feature_mapping(X_in,2);
X_train = [ones(n,1) X_map];

% Use active learning to choose data points
idx=active_learn(X_train,k1,k2);
plot_points(X_map,idx,setdiff(1:n,idx));

fprintf('Active learning chosen data have been circled in red; while remaining data in blue.\n');
