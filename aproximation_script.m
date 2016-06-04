clc;
clear;


net=createNet(1,1,[8],[3,1]);


%%przygotowanie danych
a=5;
for i=1:50
    Xu(i,:)=(a/50)*(i-1);
    T(i,:)=y(Xu(i,:)); %
end


 
%% 
%  train_LM( net,Xu,Yu,max_error,max_epochs )
% Xu, Yu - dane uczac
% max_error,max_epochs - warunki zakoñczenia nauki

net=train_LM(net,Xu,T,0.00001,1000);

    
%%Test i rysowanie wynikow
for i=1:1000
    X_t(i)=(a/1000)*(i-1);
end
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

