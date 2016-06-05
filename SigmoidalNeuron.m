classdef SigmoidalNeuron < BaseNeuron
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
         function obj = activation_function(obj,x)
            obj.response=tansig(x);
       end
    end
    
end

