clc
cost = zeros(5,1);
costc = zeros(5,1);
costt = zeros(5,1);
alpha = 0.01;
iteration = 10000;
Y = target(:,1);
x1= age(:,1);
x2 = chol(:,1);
x3 = cp(:,1);
x4 = fbs(:,1);
x5 = thal(:,1);
x6 = oldpeak(:,1);
x7 = sex(:,1);
x8 = ca(:,1);
Yall= Y;
Y = Yall(1:200,:);
Yc = Yall(201:225,:);
Yt = Yall(226:250,:);

U = [ones(length(x1),1),x1,x2,x3,x4,x5,x8];
U = normalize(U);
U2 = [ones(length(x1),1),x1,x2.^2,x3.^3,x4.^4,x5.^5];
U2 = normalize(U2);
U3 = [ones(length(x1),1),x1,x2,x3,x4,x5,x1.^2,x2.^2,x3.^2];
U3 = normalize(U3);
U4 = [ones(length(x1),1),x1,x2,x3,x4,x5,x6,x7,x8,x1.^2,x2.^2,x3.^2,x4.^2,x5.^2];
U4 = normalize(U4);
U5 = [ones(length(x1),1),exp(x1),exp(x2),exp(x3),exp(x4),exp(x5),exp(x6),exp(x7),x1.^2,x2.^2,x3.^2,x4.^2,x5.^2,x1.^3,x2.^3,x3.^3,x4.^3,x5.^3];
U5 = normalize(U5);

%Hyp1
H1 = U(1:200,:); %training data
Hc1 = U(201:225,:); %validation data
Ht1 = U(226:250,:); %testing data
theta1 = zeros(length(H1(1,:)),1);
size(H1)

[ theta, Jvect ] = logGradient(H1,Y,theta1,alpha,iteration);
cost(1) = logcostfunction(H1,Y,theta) %cost of training data
costc(1) = logcostfunction(Hc1,Yc,theta) %cost of validation data data
costt(1) = logcostfunction(Ht1,Yt,theta) %cost of testing data data
figure(1) 
plot (Jvect)
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')

%Hyp2
H2 = U2(1:200,:); %training data
Hc2 = U2(201:225,:); %validation data
Ht2 = U2(226:250,:); %testing data

theta2 = zeros(length(H2(1,:)),1);

[ theta2, Jvect2 ] = logGradient(H2,Y,theta2,alpha,iteration);
cost(2) = logcostfunction(H2,Y,theta2)
costc(2) = logcostfunction(Hc2,Yc,theta2) %cost of validation data data
costt(2) = logcostfunction(Ht2,Yt,theta2) %cost of testing data data
figure(2) 
plot (Jvect2)
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')

%Hyp 3

H3 = U3(1:200,:); %training data
Hc3 = U3(201:225,:); %validation data
Ht3 = U3(226:250,:); %testing data
theta3 = zeros(length(H3(1,:)),1);

[ theta3, Jvect3 ] = logGradient(H3,Y,theta3,alpha,iteration);
cost(3) = logcostfunction(H3,Y,theta3)
costc(3) = logcostfunction(Hc3,Yc,theta3) %cost of validation data data
costt(3) = logcostfunction(Ht3,Yt,theta3) %cost of testing data data
figure(3) 
plot (Jvect3)
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')

%Hyp 4
H4 = U4(1:200,:); %training data
Hc4 = U4(201:225,:); %validation data
Ht4 = U4(226:250,:); %testing data
theta4 = zeros(length(H4(1,:)),1);

[ theta4, Jvect4 ] = logGradient(H4,Y,theta4,alpha,iteration);
cost(4) = logcostfunction(H4,Y,theta4)
costc(4) = logcostfunction(Hc4,Yc,theta4) %cost of validation data data
costt(4) = logcostfunction(Ht4,Yt,theta4) %cost of testing data data
figure(4) 
plot (Jvect4)
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')

%Hyp 5
H5 = U5(1:200,:); %training data
Hc5 = U5(201:225,:); %validation data
Ht5 = U5(226:250,:); %testing data
theta5 = zeros(length(H5(1,:)),1);

[ theta5, Jvect5 ] = logGradient(H5,Y,theta5,alpha,iteration);
cost(5) = logcostfunction(H5,Y,theta5)
costc(5) = logcostfunction(Hc5,Yc,theta5) %cost of validation data data
costt(5) = logcostfunction(Ht5,Yt,theta5) %cost of testing data data
figure(5) 
plot (Jvect5)
xlabel('Number of iterations') 
ylabel('Error') 
title('Cost function')


%The required optimal thetas
theta;
theta2;
theta3;
theta4;

figure(6)
scatter(1:1:5,cost,'r')
hold on
scatter(1:1:5,costc,'*')
hold on 
scatter (1:1:5,costt,'g')
hold off
legend('Training data','Cross Validation Cost','Testing Cost')
xlabel('Hypothesis Numbeer') 
ylabel('Error') 
title('Comparing Hypothesis')

% From the figure we conclude that the 4th hyp is the most accurate hyp
