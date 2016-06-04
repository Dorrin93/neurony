function net = createNet( inputNumber,outputNumber,hiddenLayersNeuronNumber, neuronTypes)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
net = FeedForwardNetwork;
net.inputLayer=inputNumber;
net.hiddenLayer=cell(1,size(hiddenLayersNeuronNumber,2));
initial_weights=0.1;
for i=1:size(hiddenLayersNeuronNumber,2)
    if i==1
        inputs=inputNumber;
    else
        inputs=hiddenLayersNeuronNumber(i-1);
    end
    for j=1:hiddenLayersNeuronNumber(i)
        net.hiddenLayer{i}(j)=getNeuron(neuronTypes(i));
        net.hiddenLayer{i}(j).weights=rand()/10* ones(1, inputs);
         net.hiddenLayer{i}(j).activation_function_parameters=[1,0];
    end

end
inputs=hiddenLayersNeuronNumber(size(hiddenLayersNeuronNumber,2));

for j=1:outputNumber
    if(j==1)
        net.outputLayer=getNeuron(neuronTypes(size(hiddenLayersNeuronNumber,2)+1));
    else
        net.outputLayer(j)=getNeuron(neuronTypes(size(hiddenLayersNeuronNumber,2)+1));
    end
        net.outputLayer(j).weights=rand()/10 * ones(1, inputs);
        net.outputLayer(j).activation_function_parameters=[1,0];
    
end
end
