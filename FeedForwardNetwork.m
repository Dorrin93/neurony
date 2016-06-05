classdef FeedForwardNetwork
    %FeedForwardNetwork
  
    properties
        inputLayer;
        hiddenLayer;
        outputLayer;
        
        transferfcns;
        FFNeuronOptions;
        
    end
    
    properties(Access = private)
        init;
        pn; ps;
        tn; ts;
        hiddenLayerSize;
    end
    
    properties(Constant)
        neuronTypes = {'Empty', 'Lin', 'Fuzzy', 'Tansig'};
    end
    
    %% Public methods
    methods
        
        function obj = FeedForwardNetwork(layers, varargin)
            if length(varargin) < size(layers,2)+1
                error('Wrong input argument size');
            end
            obj.hiddenLayerSize = layers;
            
            obj.transferfcns = cell(1, size(layers,2)+1);
            for i=1:size(layers,2)+1
                if any(strcmp(obj.neuronTypes, varargin{i}))
                    obj.transferfcns{i} = varargin{i};
                else
                    error('Wrong transfer function name');
                end
            end
            
            fuzzies = sum(ismember(obj.transferfcns,'Fuzzy'));
            obj.FFNeuronOptions = cell(fuzzies,1);
            for i=1:fuzzies
                obj.FFNeuronOptions{i} = {'D','Hamacher'};
            end
            
            obj.init = false;
            
        end
        
        function obj = configure(obj, X, Y)
            is = size(X, 2);
            os = size(Y, 2);
            hs = size(obj.transferfcns,2)-1;
            [obj.pn, obj.ps] = mapminmax(X');
            [obj.tn, obj.ts] = mapminmax(Y');
            
            obj.inputLayer=is;
            obj.hiddenLayer=cell(1,hs);
            fuzzyiter = 1;
            for i=1:hs
                if i==1
                    inputs=is;
                else
                    inputs=obj.hiddenLayerSize(i-1);
                end
                
                for j=1:obj.hiddenLayerSize(i)
                    obj.hiddenLayer{i}(j) = getNeuron(obj, i, fuzzyiter);
                    obj.hiddenLayer{i}(j).weights=rand()/10* ones(1, inputs);
                    obj.hiddenLayer{i}(j).activation_function_parameters=[1,0];
                end
                
                if strcmp(obj.transferfcns{i},'Fuzzy')
                    fuzzyiter = fuzzyiter + 1;
                end          
            end
            
            inputs=obj.hiddenLayerSize(end);
            
            for j=1:os
                if(j==1)
                    obj.outputLayer=getNeuron(obj, hs+1, fuzzyiter);
                else
                    obj.outputLayer(j)=getNeuron(obj, hs+1, fuzzyiter);
                end
                obj.outputLayer(j).weights=rand()/10 * ones(1, inputs);
                obj.outputLayer(j).activation_function_parameters=[1,0];              
            end
            
            obj.init = true;            
            
        end
        
        function net_response = calculate_output(obj,X)
            %x = mapminmax('apply',X', obj.ps);
            x = X;
            
            %%HiddenLayers
            input=x';
            for i=1:size(obj.hiddenLayer,2)
                hiddenLayerOutput=zeros(1,size(obj.hiddenLayer{i},2));
                for j=1:size(obj.hiddenLayer{i},2)
                    hiddenLayerOutput(j)=obj.hiddenLayer{i}(j).calculate_output(input).response;
                end
                input=hiddenLayerOutput;
            end
            %%OutputLayer
            response=zeros(1,size(obj.outputLayer,2));
            for i=1:size(obj.outputLayer,2)
                response(i)=obj.outputLayer(i).calculate_output(hiddenLayerOutput).response;
            end
            net_response = mapminmax('reverse', response', obj.ts)';
        end
        
        function net = train_LM( net,Xu,Yu,max_error,max_epochs,max_mu )
            %Function for training network with Levenberg-Marquardt Backpropagation
            %Algorithm
            %Xu - learning inputs
            %Yu - learning outputs
            mu=0.1;
            old_err=999999;
            m=1;
            
            %x = mapminmax('apply',Xu', net.ps)';
            %y = mapminmax('apply',Yu', net.ps)';
            x = Xu;
            y = Yu;
            
            for i=1:max_epochs
                net2=net;
                rp=randperm(size(x,1));
                x=x(rp,:);
                y=y(rp,:);
                net2=LM_iteration(net2,x,y,mu);
                new_err=0;
                %%blad
                for it=1: size(x,1)
                    new_err=new_err+(y(it,:)-net2.calculate_output(x(it,:)))^2;
                end
                %%aktualizacja jesli blad sie zmniejszyl
                if(old_err-new_err>0.000001)
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
                if(mu>max_mu)
                    break
                end
            end
        end
        
        function net = train_BMAM( net,Xu,Yu,max_error,mu )
            %%Function for training network with BMAM
            old_err=0;
            
            %x = mapminmax('apply',Xu', net.ps)';
            %y = mapminmax('apply',Yu', net.ps)';
            x = Xu;
            y = Yu;
            
            for it=1: size(x,1)
                old_err=old_err+(y(it,:)-net.calculate_output(x(it,:)))^2;
            end
            rp=randperm(size(x,1));
            x=x(rp,:);
            y=y(rp,:);
            for i=1:size(net.hiddenLayer,2)
                for j=1:size(net.hiddenLayer{i},2)
                    for weight=1:size(net.hiddenLayer{i}(j).weights,2)+1
                        
                        new_err=[];
                        lowest_err=0;
                        for copy=1:6 %% tutaj zrownoleglenie
                            net_copies(copy)=net;
                            if copy<6 %% numer 6 pozostaje bez zmian
                                
                                if(weight==size(net.hiddenLayer{i}(j).weights,2)+1)
                                    net_copies(copy).hiddenLayer{i}(j).bias=net_copies(copy).hiddenLayer{i}(j).bias+(.5-rand());
                                else
                                    net_copies(copy).hiddenLayer{i}(j).weights(weight)=net_copies(copy).hiddenLayer{i}(j).weights(weight)+(.5-rand());
                                end
                            end
                            for iter=1:3
                                net_copies(copy)=LM_iteration(net_copies(copy),x,y,mu);
                            end
                            new_err(copy)=0;
                            for it=1: size(x,1)
                                new_err(copy)=new_err(copy)+(y(it,:)-net_copies(copy).calculate_output(x(it,:)))^2;
                            end
                        end
                        
                        [min_err,min_err_index]=min(new_err);
                        %%aktualizacja jesli blad sie zmniejszyl
                        if(old_err-min_err>0.00001)
                            net=net_copies(min_err_index);
                            old_err=new_err(min_err_index);
                            mu=mu/10;
                            
                        else
                            
                            mu=mu*10;
                        end
                        old_err
                        
                        if(old_err<max_error)
                            break
                        end
                    end
                    
                    j
                end
                i
            end
            for j=1:size(net.outputLayer,2)
                for weight=1:size(net.outputLayer(j).weights,2)+1
                    for copy=1:6 %% tu tez zrownolglic mozna
                        net_copies(copy)=net;
                        if copy<6 %%numer szesc bez zmian
                            if(weight==size(net.outputLayer(j).weights,2)+1)
                                net_copies(copy).outputLayer(j).bias=net_copies(copy).outputLayer(j).bias+(.5-rand());
                            else
                                net_copies(copy).outputLayer(j).weights(weight)=net_copies(copy).outputLayer(j).weights(weight)+(.5-rand());
                            end
                        end
                        for iter=1:3
                            net_copies(copy)=LM_iteration(net_copies(copy),x,y,mu);
                        end
                        new_err(copy)=0;
                        for it=1: size(x,1)
                            new_err(copy)=new_err(copy)+(y(it,:)-net_copies(copy).calculate_output(x(it,:)))^2;
                        end
                    end
                    [min_err,min_err_index]=min(new_err);
                    %%aktualizacja jesli blad sie zmniejszyl
                    if(old_err-min_err>0.00001)
                        net=net_copies(min_err_index);
                        old_err=new_err(min_err_index);
                        mu=mu/10;
                        
                    else
                        
                        mu=mu*10;
                    end
                    
                    old_err
                    
                    if(old_err<max_error)
                        break
                    end
                end
                j
            end
            
        end
        
    end
    %% Public methods end
    
    %% Privare methods
    methods(Access = private)
        
        function [net_response,y,delta] = calculate_output_and_LMparameters(obj,X)
            dx=0.00001;
            %%HiddenLayers
            
            y=cell(1,size(obj.hiddenLayer,2)+1);
            s=cell(1,size(obj.hiddenLayer,2)+1);
            delta=cell(1,size(obj.hiddenLayer,2)+1);
            input=X;
            for i=1:size(obj.hiddenLayer,2)
                hiddenLayerOutput=zeros(1,size(obj.hiddenLayer{i},2));
                s{i}=zeros(1,size(obj.hiddenLayer{i},2));
                y{i}=zeros(1,size(input,2)+1);
                y{i}=[input 1];
                for j=1:size(obj.hiddenLayer{i},2)
                    neti=obj.hiddenLayer{i}(j).calculate_net(input);
                    s{i}(j)=(obj.hiddenLayer{i}(j).activation_function(neti+dx).response-obj.hiddenLayer{i}(j).activation_function(neti-dx).response)/(2*dx);
                    hiddenLayerOutput(j)=obj.hiddenLayer{i}(j).calculate_output(input).response;
                end
                input=hiddenLayerOutput;
            end
            y{size(obj.hiddenLayer,2)+1}=[input 1];
            %%OutputLayer
            net_response=zeros(1,size(obj.outputLayer,2));
            s{size(obj.hiddenLayer,2)+1}=zeros(1,size(obj.outputLayer,2));
            delta{size(obj.hiddenLayer,2)+1}=zeros(size(obj.outputLayer,2),size(obj.outputLayer,2));
            for i=1:size(obj.outputLayer,2)
                neti=obj.outputLayer(i).calculate_net(input);
                s{size(obj.hiddenLayer,2)+1}(i)=(obj.outputLayer(i).activation_function(neti+dx).response-obj.outputLayer(i).activation_function(neti-dx).response)/(2*dx);
                net_response(i)=obj.outputLayer(i).calculate_output(hiddenLayerOutput).response;
                
                delta{size(obj.hiddenLayer,2)+1}(i,i)=s{size(obj.hiddenLayer,2)+1}(i);
            end
            %%BACKPROP
            delta{size(obj.hiddenLayer,2)}=zeros(size(obj.outputLayer,2),size(obj.hiddenLayer{size(obj.hiddenLayer,2)},2));
            for j=1:size(obj.outputLayer,2)
                for k=1:size(obj.hiddenLayer{size(obj.hiddenLayer,2)},2)
                    delta{size(obj.hiddenLayer,2)}(j,k)=obj.outputLayer(j).weights(k)*delta{size(obj.hiddenLayer,2)+1}(j,j) *s{size(obj.hiddenLayer,2)}(k);
                end
            end
            for i=size(obj.hiddenLayer,2)-1:-1:1
                delta{i}=zeros(size(obj.hiddenLayer{i+1},2),size(obj.hiddenLayer{i},2));
                
                for j=1:size(obj.outputLayer,2)
                    for k=1:size(obj.hiddenLayer{i},2)
                        for i2=1:size(obj.hiddenLayer{i+1},2)
                            
                            % j
                            
                            delta{i}(j,k)=delta{i}(j,k)+obj.hiddenLayer{i+1}(j).weights(i2)*delta{i+1}(j,i2);
                        end
                        delta{i}(j,k)=delta{i}(j,k)*s{i}(k);
                    end
                end
                
            end
        end
        
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


        function neuron = getNeuron(obj, layerIndex, FFoptionIndex)
            switch obj.transferfcns{layerIndex}
                case 'Empty'
                    neuron = EmptyNeuron;
                case 'Lin'
                    neuron = LinearNeuron;
                case 'Tansig'
                    neuron = SigmoidalNeuron;
                case 'Fuzzy'
                    neuron = FFNeuron(obj.FFNeuronOptions{FFoptionIndex}{:});
            end
        end 
    end
    %% Private methods end
    
end

