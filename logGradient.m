function [ theta, Jvect ] = logGradient( X,Y,theta,alpha,iteration )

[n m]=size(X);
Jvect = zeros(iteration,1);

for i=1:iteration
    h = sigmoid(X * theta)
   % thetax = theta (2:m,1);
    %+(0.01/(2*m))*sum(thetax.^2);
    theta=theta*(1-(alpha*0.0003/m))-(alpha/m)*X'*(h-Y);
    Jvect(i) = logcostfunction(X,Y,theta);
    
end

