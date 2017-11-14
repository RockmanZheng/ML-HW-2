function idx=randomly_select_impl(X,idx_now)
    % X is current data set in query
    % Goal: find the next data point in X randomly
    
    [n,~]=size(X);
    % Get complement indices
    % In other words, we choose from data that is not yet queried
    idx_comp=setdiff(1:n,idx_now);
    idx=idx_comp(ceil(rand()*length(idx_comp)));
end