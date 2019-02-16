function J = costfunction( X,Y,theta )

m=length(X);
f = length(theta);

h = X * theta;

J = (1/(2*m)) * (sum((h-Y).^2))+ ((0.01/(2*m)* sum((theta(2:f,1)).^2)));

end

