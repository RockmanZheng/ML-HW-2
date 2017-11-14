function generate_a1_data(n)
    X_in = random('Normal',0,2,n,3);
    theta = [0.3 -0.4 1.7];
    bias = 10;
    y_true = X_in*theta'+bias;
    y_noisy = y_true+random('Normal',0,1,n,1);
    save('a1_data','X_in','y_true','y_noisy');

end