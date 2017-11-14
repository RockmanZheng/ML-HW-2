function [alpha,error_count] = train_kernel_perceptron(X,y,kernel)
    [n,~]=size(X);
%     alpha = rand(n,1);
    alpha = rand(n,1)-0.5;
    error_count = 0;
    % 首先找到第一个错误分类的数据点，更新一次，计数错误分类个数
    for i=1:n
        if y(i)*discriminant_function(alpha,X,kernel,X(i,:))<0
            error_count = error_count+1;
            alpha(i) = alpha(i)+y(i);
        end
    end
    error_temp = error_count;
    steady_count = 0;
    k=10;
    error_rates = ones(1,k);
    epsilon = 1e-5;
    % 然后不断迭代，直到全部样本都正确区分
    while error_count~=0
        error_count = 0;
        for i=1:n
            if y(i)*discriminant_function(alpha,X,kernel,X(i,:))<0
                error_count = error_count+1;
                alpha(i) = alpha(i)+y(i);
            end
        end
        % If error_count does not go down in 5 steps, we think the
        % algorithm has already converge
        
        % Remark, the convergence behaviour of training varies with
        % different kernel
        % If using radial basis kernel, the training error will decrease
        % monotonically; but if with polynomial kernel, the training error
        % may fluctuate slightly in a steady manner.
        % In order to take these 2 situations into consideration, a new
        % convergence test is needed.
        if error_temp~=error_count
            steady_count = 0;
        else
            steady_count = steady_count+1;
        end
        if steady_count>k
            break;
        end
        error_temp = error_count;
        % Update last k error rates
        error_rates = [error_rates(2:end) error_count/n];
        % If last k error rates do not vary much, then we think it
        % has converged
        if var(error_rates)<epsilon
            break;
        end
    end
end