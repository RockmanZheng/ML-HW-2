function ax=plot_dec_surf(Range,nSlices,X_train,alpha,kernel)
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
            Z(i,j)=discriminant_function(alpha,X_train,kernel,[X(i,j) Y(i,j)]);
        end
    end
    figure
    s=surf(X,Y,Z);
    shading interp
    xlabel('X');
    ylabel('Y');
    zlabel('Discriminant Function');
    title('Decision Surface');
    ax = s.Parent;
end