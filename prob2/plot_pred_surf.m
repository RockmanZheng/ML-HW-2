function ax=plot_pred_surf(Range,nSlices,X_train,alpha,kernel)
    % Retrieve plot range
    xmin = Range(1,1);
    xmax = Range(1,2);
    ymin = Range(2,1);
    ymax = Range(2,2);
    % Retrieve mesh slices
    x_nSlices = nSlices(1);
    y_nSlices = nSlices(2);
    x=linspace(xmin,xmax,x_nSlices);
    y=linspace(ymin,ymax,y_nSlices);
    [X,Y]=meshgrid(x,y);
    [m,n]=size(X);
    Z=zeros(m,n);
    % Compute surface
    for i=1:m
        for j=1:n
            Z(i,j)=kernel_pred([X(i,j) Y(i,j)],X_train,alpha,kernel);
        end
    end
    figure;
%     ax = fig.Axes;
%     title('Prediction Surface');
    s=surf(X,Y,Z);
    shading interp;
    hold on
    ax=s.Parent;
end