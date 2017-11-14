function y_hat=kernel_pred(X,X_train,alpha,kernel)
    y_hat=kernel(X,X_train)*alpha;
end