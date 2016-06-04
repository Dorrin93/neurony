clc;
clear;
max_epoch=500;
mu=100;
net=createNet(1,1,[4,4],[2,2,1]);
% Xu=[1;2;3;4];
% T=[1;3;5;7];
old_err=999999;
a=5;
%%przygotowanie danych
for i=1:50
    Xu(i,:)=(a/50)*(i-1);
    T(i,:)=y(Xu(i,:));
end
m=1;

 
for i=1:1000
    X_t(i)=(a/1000)*(i-1);
end
net=train_LM(net,Xu,T,0.00001,1000);
N=[];
    
    d=[];
    for i=1:1000
        T_u(i)=net.calculate_output(X_t(i));
        d=[d y(X_t(i))];
        i
    end
   
        sum(abs(T_u-(d)))/1000
   plot(X_t,d,'b');
   hold on;
   plot(X_t,T_u,'rp');

