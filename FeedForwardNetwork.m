classdef FeedForwardNetwork
    %FeedForwardNetwork
    
    
    properties
        inputLayer;
        hiddenLayer;
        outputLayer;
    end
    
    methods
        function obj = FeedForwardNetwork()
            %EMPTY
            
            
        end
        function net_response = calculate_output(obj,X)
            %%HiddenLayers
            input=X;
            for i=1:size(obj.hiddenLayer,2)
                hiddenLayerOutput=zeros(1,size(obj.hiddenLayer{i},2));
                for j=1:size(obj.hiddenLayer{i},2)
                    hiddenLayerOutput(j)=obj.hiddenLayer{i}(j).calculate_output(input).response;
                end
                input=hiddenLayerOutput;
            end
            %%OutputLayer
            net_response=zeros(1,size(obj.outputLayer,2));
            for i=1:size(obj.outputLayer,2)
                net_response(i)=obj.outputLayer(i).calculate_output(hiddenLayerOutput).response;
            end
        end
        %%
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
        function net_response=change_weights_and_calculate_output(obj,X,w)
            net=obj;
            weight_num=1;
            for i=1:size(net.hiddenLayer,2)
                for j=1:size(net.hiddenLayer{i},2)
                    for k=1:size(net.hiddenLayer{i}(j),2)
                        net.hiddenLayer{i}(j).weights(k)=w(weight_num);
                        weight_num=weight_num+1;
                    end
                    net.hiddenLayer{i}(j).bias=w(weight_num);
                    weight_num=weight_num+1;
                end
            end
            for j=1:size(obj.outputLayer,2)
                for k=1:size(obj.hiddenLayer(j),2)
                    net.outputLayer(j).weights(k)=w(weight_num);
                    weight_num=weight_num+1;
                end
                net.outputLayer(j).bias=w(weight_num);
                weight_num=weight_num+1;
            end
            %%HiddenLayers
            input=X;
            for i=1:size(net.hiddenLayer,2)
                hiddenLayerOutput=zeros(1,size(net.hiddenLayer{i},2));
                for j=1:size(net.hiddenLayer{i},2)
                    hiddenLayerOutput(j)=net.hiddenLayer{i}(j).calculate_output(input);
                end
                input=hiddenLayerOutput;
            end
            %%OutputLayer
            net_response=zeros(1,size(net.outputLayer,2));
            for i=1:size(net.outputLayer,2)
                net_response(i)=net.outputLayer(i).calculate_output(hiddenLayerOutput);
            end
        end
    end
    
end

