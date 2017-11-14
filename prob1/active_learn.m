function idx=active_learn(X,k1,k2)
    idx = zeros(1,k1+k2);
    
    [n,~]=size(X);
    % Randomly pick first k1 data points
    for i=1:k1
       idx(i)=ceil(rand()* n);
    end
    
%     idx(1:k1)=1:k1;
    % Query data points one by one
    for i=1:k2
        idx(k1+i)=active_learn_impl(X,idx(idx>0));
    end
end