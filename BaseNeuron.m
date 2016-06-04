classdef BaseNeuron
    %   BaseNeuron - base class for other neurons
    %   It includes only input weights, c
    
    properties
        weights;
        activation_function_parameters;
        bias;
        response;
    end
    
    methods
        function obj = BaseNeuron()
            %EMPTY and should stay that way
          %  obj=BaseNeuron;
            obj.bias=1;
        end
        function obj=set.weights(obj,input_weights)
           % obj=BaseNeuron;
            obj.weights=input_weights;
            
        end
        %%default activation function, it exsist solely for the sake of be
        %%overloaded ;_;
        function out = activation_function(obj,x)
            out=x;
        end
        
        function response = calculate_output(obj,X)
            %%calculate neuron response for input vector X
            response = obj.activation_function(sum(X.*obj.weights)+obj.bias);
        end
        function response = calculate_net(obj,X)
            %input=sum(X.*weights)+bias;
            response = sum(X.*obj.weights)+obj.bias;
        end
    end
    
end

