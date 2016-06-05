classdef BaseNeuron < handle
    %   BaseNeuron - base class for other neurons
    %   It includes only input weights, c
    
    properties
        weights;
        activation_function_parameters;
        bias;
    end
    
    methods
        function obj = BaseNeuron()
            % EMPTY and should stay that way
            obj@handle();
            obj.bias=1;
        end
        function obj=set.weights(obj,input_weights)
            obj.weights=input_weights;          
        end
        %% default activation function, it exsist solely for the sake of be
        %  overloaded ;_;
        function response = activation_function(obj,x)
            response = x;
        end
        
        function response = calculate_output(obj,X)
            %% calculate neuron response for input vector X
            response = obj.activation_function(sum(X.*obj.weights)+obj.bias);
        end
        function response = calculate_net(obj,X)
            response = sum(X.*obj.weights)+obj.bias;
        end
    end
    
end

