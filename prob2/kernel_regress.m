function alpha=kernel_regress(X,y,kernel,lambda)
    n=length(y);
    if lambda==0
        alpha = pinv(kernel(X,X))*y;
    else
        alpha=linsolve(lambda*eye(n)+kernel(X,X),y);
    end
end