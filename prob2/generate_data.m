% Generate data from mixed Gaussian function

% Height of Gaussian function
h1=10;
h2=-8;
h3=12;
% Center of Gaussian function
mu1=[10 12];
mu2=[-8 20];
mu3=[30 -28];
% Variance of Gaussian function
var1=349;
var2=316;
var3=325;
F1=@(x)(h1*exp(-sum((x-mu1).^2,2)/var1/2));
F2=@(x)(h2*exp(-sum((x-mu2).^2,2)/var2/2));
F3=@(x)(h3*exp(-sum((x-mu3).^2,2)/var3/2));
F=@(x)(F1(x)+F2(x)+F3(x));

n=1000;
d=2;
X=(rand(n,d)-0.5)*100;
y_true = F(X);
figure
scatter3(X(:,1),X(:,2),y_true);
sigma = 2;
y_noisy = y_true+randn(size(y_true))*sigma;
figure
scatter3(X(:,1),X(:,2),y_noisy);
save('fake_data','X','y_true','y_noisy');

