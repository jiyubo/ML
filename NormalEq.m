function [ Newtheta ] = NormalEq( X,Y )
L = (transpose(X)) *X
l = (inv(L));
f= (transpose(X))*Y;
Newtheta = l*f;

end

