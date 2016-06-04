function net = train_LM( net,Xu,Yu,max_error,max_epochs )
%Function for training network with Levenberg-Marquardt Backpropagation
%Algorithm
%Xu - learning inputs
%Yu - learning outputs
mu=0.1;
old_err=999999;
m=1;


for i=1:max_epochs
    net2=net;
    rp=randperm(size(Xu,1));
    Xu=Xu(rp,:);
    Yu=Yu(rp,:);
    net2=LM_iteration(net2,Xu,Yu,mu);
    new_err=0;
    %%blad
    for it=1: size(Xu,1)
        new_err=new_err+(Yu(it,:)-net2.calculate_output(Xu(it,:)))^2;
    end
    %%aktualizacja jesli blad sie zmniejszyl
    if(old_err-new_err>0.00001)
        net=net2;      
        old_err=new_err;     
        mu=mu/10;
        m=1;
    else
        m=m+1;
        mu=mu*10;
    end
    
    old_err
    i
    if(old_err<max_error)
        break
    end
    if(mu>10000000000)
        break
    end
end



end

