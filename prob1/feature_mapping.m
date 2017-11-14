function [X_map]= feature_mapping(X,mode)
    if mode==1
        X_map=X;
    elseif mode==2
        X_map=log(X.^2);
    else
        X_map=X;
    end
end