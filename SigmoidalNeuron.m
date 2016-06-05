classdef SigmoidalNeuron < BaseNeuron
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
         function response = activation_function(obj,x)
            response=tansig(x);
       end
    end
    
end

