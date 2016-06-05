function net = trainFPA(net, X, Y, n, MaxGeneration, p)
weightNum = 0;
for i = 1:length(net.hiddenLayer)
    for j = 1:length(net.hiddenLayer{i})
        weightNum = weightNum + 1 + length(net.hiddenLayer{i}(j).weights);
    end;
end;
weightNum = weightNum+ 1 + length(net.outputLayer.weights);
ampli = 1e1;
weights =ampli* (rand(n,weightNum)-ones(n,weightNum)*.5);

errors = zeros(n,1);
for k = 1:n;
    idx = 0;
    for i = 1:length(net.hiddenLayer)
        for j = 1:length(net.hiddenLayer{i})
            idx = idx+1;
            net.hiddenLayer{i}(j).bias = weights(idx);
            for l=1:length(net.hiddenLayer{i}(j).weights);
                idx = idx+1;
                net.hiddenLayer{i}(j).weights(l) = weights(idx);
            end;
        end;
    end;
    idx = idx+1;
    net.outputLayer.bias = weights(idx);
    for i=1:length(net.outputLayer.weights)
        idx = idx+1;
        net.outputLayer.weights(i) = weights(idx);
    end;
    for it=1: size(X,1)
        errors(k)=errors(k)+(Y(it,:)-net.calculate_output(X(it,:)))^2;
    end
end;

[v idx] = min(errors);
g = weights(idx,:);


for t= 1:MaxGeneration
    t
    for k = 1:n
        if rand() < p
            L = Levy(weightNum);
            new_weights = weights(k,:)+L.*(g-weights(k,:));
        else
            e = rand(1,weightNum);
            new_weights = weights(k,:)+e.*(weights(randi([1 n]),:)-weights(randi([1 n]),:));
        end;
        
        idx = 0;
        for i = 1:length(net.hiddenLayer)
            for j = 1:length(net.hiddenLayer{i})
                idx = idx+1;
                net.hiddenLayer{i}(j).bias = new_weights(idx);
                for l=1:length(net.hiddenLayer{i}(j).weights);
                    idx = idx+1;
                    net.hiddenLayer{i}(j).weights(l) = new_weights(idx);
                end;
            end;
        end;
        
        idx = idx+1;
        net.outputLayer.bias = weights(idx);
        for i=1:length(net.outputLayer.weights)
            idx = idx+1;
            net.outputLayer.weights(i) = weights(idx);
        end;

        err = 0;
        for it=1: size(X,1)
            err=err+(Y(it,:)-net.calculate_output(X(it,:)))^2;
        end;
        if errors(k) > err
            errors(k) = err;
            weights(k,:) = new_weights;
        end;
    end;
    [v idx] = min(errors);
    v
    g = weights(idx,:)
end;

idx = 0;
for i = 1:length(net.hiddenLayer)
    for j = 1:length(net.hiddenLayer{i})
        idx = idx+1;
        net.hiddenLayer{i}(j).bias = g(idx);
        for l=1:length(net.hiddenLayer{i}(j).weights);
            idx = idx+1;
            net.hiddenLayer{i}(j).weights(l) = g(idx);
        end;
    end;
end;
idx = idx+1;
net.outputLayer.bias = weights(idx);
for i=1:length(net.outputLayer.weights)
    idx = idx+1;
    net.outputLayer.weights(i) = weights(idx);
end;
end

