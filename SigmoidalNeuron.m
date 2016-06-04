classdef SigmoidalNeuron < BaseNeuron
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
         function response = activation_function(obj,x)
            obj.response=(2/(1+exp(-2*x)))-1;
            response=obj;
        end
    end
    
end

