function plot_points(X,idx_r,idx_b)
    % Goal: plot each row of X as a point in R^3. The points indexed by
    % idx_r will be circled in red and those marked by idx_b will be
    % circled (larger) in blue.
    figure
    scatter3(X(idx_r,1),X(idx_r,2),X(idx_r,3),20,'r');
    hold on
    scatter3(X(idx_b,1),X(idx_b,2),X(idx_b,3),40,'b');
end