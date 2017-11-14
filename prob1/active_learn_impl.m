function idx=active_learn_impl(X,idx_now)
    % X is current data set in query
    % Goal: find the next data point in X that we prefer to query
    % Idea 1: find the row in X that is closet to the eigendirection of
    % (X'*X)^-1 with the largest eigenvector (mesured in inner product)
    % Idea 2: directly compute object function (v'*A^2*v)/(1+v'*A*v) and
    % choose the largest element
    
    % Use pseudo inverse to tolerate degenerate cases
    A = pinv(X(idx_now,:)'*X(idx_now,:));

    [n,~]=size(X);
    % Eigen vector method
%     [v,~]=eigs(A,1);
%     % v is eigendirection of (X'*X)^-1 with the largest eigenvector
%     % Get complement indices
%     % In other words, we choose from data that is not yet queried
%     idx_comp=setdiff(1:n,idx_now);
%     X = X(idx_comp,:);
%     
%     % Normalize each row in X
%     X=X./sqrt(sum(X.^2,2));
%     % Compute inner product
%     Angle=X*v;
%     [~,this_idx]=max(Angle);
%     % Retrieve index
%     idx=idx_comp(this_idx);
%     

    % Direct method
    % Get complement indices
    % In other words, we choose from data that is not yet queried
    idx_comp=setdiff(1:n,idx_now);
    X = X(idx_comp,:);
    temp=dot(X',A^2*X')./(1+dot(X',A*X'));
    [~,this_idx]=max(temp);
    idx=idx_comp(this_idx);
end