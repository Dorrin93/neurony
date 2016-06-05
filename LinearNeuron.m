classdef LinearNeuron < BaseNeuron
    %LinearNeuron simple neuron with linear activation function
    
    properties
    end
    
    methods
        function response = activation_function(obj,x)
            response=obj.activation_function_parameters(1)*x+obj.activation_function_parameters(2);
        end
    end
    
end

