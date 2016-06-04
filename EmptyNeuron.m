classdef EmptyNeuron < BaseNeuron
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function response = activation_function(obj,x)
            obj.response=x;
            response = obj;
        end
        function response = calculate_output(obj,X)
            %%calculate neuron response for input vector X
            obj.response = sum(X*obj.weights')+obj.bias;
            response=obj;
        end
    end
    
end

