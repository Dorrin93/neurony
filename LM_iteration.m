function [net,er] = LM_iteration( net,X,Y,mu )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

er=zeros(size(X,1),size(Y,2));
output=[];



hidden_layer_sizes=[];
for i=1:size(net.hiddenLayer,2)
    hidden_layer_sizes=[hidden_layer_sizes size(net.hiddenLayer{i},2)];
end
tmp=zeros(size(er(1,:),2),sum(hidden_layer_sizes+size(net.outputLayer,2)));
w=[];
for i=1:size(net.hiddenLayer,2)
    for j=1:size(net.hiddenLayer{i},2)
        w=[w net.hiddenLayer{i}(j).weights net.hiddenLayer{i}(j).bias];
    end
end
for j=1:size(net.outputLayer,2)
    w=[w net.outputLayer(j).weights net.outputLayer(j).bias ];
end

er2=[];

    er2=zeros(size(Y,2),length(w));
for sample=1:size(X,1) %p
    [out,y,delta]=net.calculate_output_and_LMparameters(X(sample,:));
    er(sample,:)= Y(sample,:)-out;
    
    for m=1:size(Y,2) %m
        w_n=1;
        for layer=1:size(delta,2) 
            for j=1:size(delta{layer},2)
                for i=1:size(y{layer},2)
                    er2((sample-1)*size(Y,2)+m,w_n)=-delta{layer}(m,j)*y{layer}(1,i);
                    w_n=w_n+1;
                end
            %    w_n=1;
            end
        end
    end
end
ii=eye(size(transpose(er2)*er2));

w=transpose(w)-((transpose(er2)*er2+ii*mu)^(-1) * (transpose(er2) *(er)));
w=transpose(w);
weight_num=1;

for i=1:size(net.hiddenLayer,2)
    for j=1:size(net.hiddenLayer{i},2)
        for k=1:size(net.hiddenLayer{i}(j).weights,2)
            net.hiddenLayer{i}(j).weights(k)=w(weight_num);
            weight_num=weight_num+1;
        end
        net.hiddenLayer{i}(j).bias=w(weight_num);
            weight_num=weight_num+1;
    end
end
for j=1:size(net.outputLayer,2)
    for k=1:size(net.outputLayer(j).weights,2)
        net.outputLayer(j).weights(k)=w(weight_num);
        weight_num=weight_num+1;
    end
    net.outputLayer(j).bias=w(weight_num);
    weight_num=weight_num+1;
end


end

