function K=rbf_kernel(X,Y,t)
    % Implement radial basis function kernel
    % Treat each row of X and Y as data points
%     t=500;
    K=exp(-pdist2(X,Y,'squaredeuclidean')/2/t);
end