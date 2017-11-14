function K = poly_kernel(X,Y,d)
    K=(1+X*Y').^d;
end