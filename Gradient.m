function [ theta, Jvect ] = Gradient( X,Y,theta,alpha,iteration )

m=12968;
Jvect = zeros(iteration,1);

for i=1:iteration
   theta = theta*(1-(alpha*0.01/m))-(alpha/m)*X'*(X*theta-Y);
    
   Jvect(i) = costfunction(X,Y,theta);
    
end

