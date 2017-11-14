fprintf('Data generator. Generate 2 bananas shaking hands and 2 circles sharing the same center\n');

% 2 bananas shaking hands
sigma1=0.3;
sigma2=0.3;
waypoints1=[3 1;1 1;1 3;3 3];
waypoints2=[2 2;4 2;4 4;2 4];
n=1000;
X_b = zeros(n,2);
figure
% class -1
X_b(1:100,:)=[rand(100,1)*2+1.5 1+randn(100,1)*sigma1];
X_b(101:400,:)=[1.5+randn(300,1)*sigma1 rand(300,1)*2+1];
X_b(401:500,:)=[rand(100,1)*2+1.5 3+randn(100,1)*sigma1];
scatter(X_b(1:500,1),X_b(1:500,2));
hold on
% class 1
X_b(501:600,:)=[rand(100,1)*2+2 2+randn(100,1)*sigma2];
X_b(601:900,:)=[4+randn(300,1)*sigma2 rand(300,1)*2+2];
X_b(901:1000,:)=[rand(100,1)*2+2 4+randn(100,1)*sigma2];
y_b=[-1*ones(1,500) ones(1,500)]';

scatter(X_b(501:1000,1),X_b(501:1000,2));

% Shuffle data
shuffle_idx = randperm(n);
X_b = X_b(shuffle_idx,:);
y_b = y_b(shuffle_idx);

% 2 circles sharing the same center
r1 = 1;
r2 = 2;
sigma1 = 0.3;
sigma2=0.3;
theta = linspace(0,pi*7,500)';
X_c= zeros(n,2);
% class 1
R1 = r1+randn(500,1)*sigma1;
X_c(1:500,:) = [R1.*cos(theta) R1.*sin(theta)];
% class 2
R2 = r2+randn(500,1)*sigma2;
X_c(501:1000,:) = [R2.*cos(theta) R2.*sin(theta)];
y_c=[-1*ones(1,500) ones(1,500)]';


figure
scatter(X_c(1:500,1),X_c(1:500,2));
hold on
scatter(X_c(501:1000,1),X_c(501:1000,2));

% Shuffle data
shuffle_idx = randperm(n);
X_c = X_c(shuffle_idx,:);
y_c = y_c(shuffle_idx);

save('fake_data','X_b','y_b','X_c','y_c');