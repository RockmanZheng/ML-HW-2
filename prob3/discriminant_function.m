function f = discriminant_function(alpha,X_train,kernel,X)
    f=kernel(X,X_train)*alpha;
end