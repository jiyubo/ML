%Run this script after the cross validation proof 

clc

error = zeros(1,5);
erorcross = zeros(1,5);

x2 = bathrooms(:,1);
x3= condition(:,1);
x4 = sqft_living(:,1);
x5 = sqft_living15(:,1);
x6 = floors(:,1);
x7 = yr_built(:,1);
x8 = yr_renovated(:,1);
Y = price (:,1);
Ynormeq = Y; %used for the normal equation

x1 = [bedrooms(:,1),x2,x3,x4,x5,x7,x8];% HYP 1
h2 = [ones(length(x1),1),x1,x1.^2,x6]; % HYP 2
h3 = [ones(length(x1),1),x1,x6.^2]; % HYP 3
h4 = [ones(length(x1),1),x1.^3,x6.^2];% HYP 4
h5 = [ones(length(x1),1),exp(x2),exp(x3),x4.^2,x6.^2,x1,x8];

%Normailizing the data

h2 = normalizee(h2);
h3 = normalizee(h3);
h4 = normalizee(h4);
h5 = normalizee(h5);
Y = price (:,1)./mean(price(:,1));

Ytraining = Y(1:12968,1); %prices for the training data
Yt = Y(12968:17291,1);    %prices for the cross validation data
Ytest = Y(17291:21613,1); %prices for the testing data


%declaring variables 
iteration =10000;
alpha =0.01;
m=length(x1);
mh2 = length(h2);
mh3 = length(h3);
mh4 = length(h4);

%For the first hyp I repeated to find the most fitting degree polynomial
%this could be found in the cross validation proof script

%HYp1

X1all = [ones(m,1),x1,x1.^2];
X1all = normalizee(X1all);
X1 = X1all(1:12968,:);
Xt1 = X1all(12968:17291,:);
Xtesting =  X1all(17291:21613,:);
theta1 = zeros(length(X1(1,:)),1);

%for evaluating the normal equation optimal thetas

% Normaltheta1 = NormalEq(X1,Y);
% hnormal = X1*Normaltheta1;
% cost1normal = (1/2*m)*sum((hnormal - Y).^2)

[theta1,Jvect] =  Gradient(X1,Ytraining,theta1,alpha,iteration); 
error(1) = costfunction(X1,Ytraining,theta1); 
erorcross(1) = costfunction(Xt1,Yt,theta1);
figure(1)
plot(1:iteration,Jvect) 
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')


% Now evaluating HYP 2
H22 = h2(1:12968,:);
thetah2 = zeros(length((H22(1,:))),1);
[thetah2,Jvecth2] =  Gradient(H22,Ytraining,thetah2,alpha,iteration);

error(2) = costfunction(h2(1:12968,:),Ytraining,thetah2) ;
erorcross(2) = costfunction(h2(12968:17291,:),Yt,thetah2);
figure(2)
plot(1:iteration,Jvecth2) 
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')


% Now evaluating HYP 3
H33 = h3(1:12968,:);
thetah3 = zeros(length(H33(1,:)),1);
[thetah3,Jvecth3] =  Gradient(h3(1:12968,:),Ytraining,thetah3,alpha,iteration);

error(3) = costfunction(H33,Ytraining,thetah3);
erorcross(3) = costfunction(h3(12968:17291,:),Yt,thetah3);
figure(3)
plot(1:iteration,Jvecth3) 
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')


% Now evaluating HYP 4
H44 = h4(1:12968,:);
thetah4 = zeros(length(H44(1,:)),1);
[thetah4,Jvecth4] =  Gradient(H44,Ytraining,thetah4,alpha,iteration);

error(4) = costfunction(h4(1:12968,:),Ytraining,thetah4);
erorcross(4) = costfunction(h4(12968:17291,:),Yt,thetah4);
figure(4)
plot(1:iteration,Jvecth4) 
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')


% Now evaluating HYP 5
H55 = h5(1:12968,:);
thetah5 = zeros(length(H55(1,:)),1);
[thetah5,Jvecth5] =  Gradient(H55,Ytraining,thetah5,alpha,iteration);

error(5) = costfunction(h5(1:12968,:),Ytraining,thetah5);
erorcross(5) = costfunction(h5(12968:17291,:),Yt,thetah5);
figure(5)
plot(1:iteration,Jvecth5) 
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')


error
erorcross

%Now to compare the hyp
figure(6) 
scatter (1:1:5,error, 'r')
hold on
scatter (1:1:5,erorcross,'*')
hold off
legend('Training data','Cross Validation Cost')
xlabel('Hypothesis Numbeer') 
ylabel('Error') 
title('Comparing Hypothesis')


%from figure 5 we can conclude that the fifth hypothesis gave the best
%cross validation result so we will test it with the testing data

% Testing data for HYP5
%costHyp1 = costfunction(Xtesting,Ytest,theta1)
costHyp5 = costfunction(h5(17291:21613,:),Ytest,thetah5)

%We conclude that Hyp number 5 is the most accurate 


