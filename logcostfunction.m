function J = logcostfunction( X,Y,theta )

m=length(X);
f = length(theta);
thetax = theta (2:f,1);
h =sigmoid((X * theta));
J = ((1/(m)) *sum((-Y.*log(h))-((1-Y).*log(1-h))))+(0.0003/(2*m))*sum(thetax.^2);
end

