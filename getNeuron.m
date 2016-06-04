function neuron = getNeuron( neuronType )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
switch neuronType
     case 0
        neuron=EmptyNeuron;
    case 1
        neuron=LinearNeuron;
        
    case 2
        neuron=FFNeuron('JK','Algebraic');
        
    case 3
        neuron=SigmoidalNeuron;
    otherwise 
        neuron=BaseNeuron;
end
    
end

