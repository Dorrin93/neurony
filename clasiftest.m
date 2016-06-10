clc;
clear;

d = 0.5;

for ds=1:3
sets = {'iris', 'wine', 'ion'};
X_t = [];
I = [];

switch ds
    case 1
    %% iris_dataset

    iris = load('iris_dataset');
    fprintf('Zbior iris, %g%% danych uczacych\n\n', d*100);

    clas = 3;
    dataPerClass = 50;
    trainSize = int16(dataPerClass * d);
    testSize = dataPerClass - trainSize;

    % przemieszanie zbioru
    X_u1 = iris.irisInputs(:,1:50);
    X_u1 = X_u1(:,randperm(50));
    X_u2 = iris.irisInputs(:,51:100);
    X_u2 = X_u2(:,randperm(50));
    X_u3 = iris.irisInputs(:,101:150);
    X_u3 = X_u3(:,randperm(50));
    I_1 = ones(1,dataPerClass);
    I_2 = ones(1,dataPerClass)*2;
    I_3 = ones(1,dataPerClass)*3;



    % losowanie zbioru uczacego
    for i = 1:trainSize
       sample = randi([1 51-i]);
       X_t = [X_t X_u1(:,sample) X_u2(:,sample) X_u3(:,sample)];
       I = [I I_1(:,sample) I_2(:,sample) I_3(:,sample)];
       X_u1(:,sample) = [];
       X_u2(:,sample) = [];
       X_u3(:,sample) = [];
       I_1(:,sample) = [];
       I_2(:,sample) = [];
       I_3(:,sample) = [];   
    end

    % zlozenie zbioru testowego
    X_s = [X_u1 X_u2 X_u3];
    I_s = [I_1 I_2 I_3];

    case 2
    %% wine_dataset
    wine = load('wine_dataset');
    fprintf('Zbior wine, %g%% danych uczacych\n\n', d*100);

    % znalezienie minimow i maximow
    class = 3;
    data1Size = 59;
    data2Size = 71;
    data3Size = 48;
    train1Size = int16(data1Size * d);
    train2Size = int16(data2Size * d);
    train3Size = int16(data3Size * d);
    trainSize = train1Size + train2Size + train3Size;
    test1Size = data1Size - train1Size;
    test2Size = data2Size - train2Size;
    test3Size = data3Size - train3Size;
    testSize = test1Size + test2Size + test3Size;

    % przemieszanie zbioru
    X_u1 = wine.wineInputs(:,1:59);
    X_u1 = X_u1(:,randperm(data1Size));
    X_u2 = wine.wineInputs(:,60:130);
    X_u2 = X_u2(:,randperm(data2Size));
    X_u3 = wine.wineInputs(:,131:178);
    X_u3 = X_u3(:,randperm(data3Size));
    I_1 = ones(1, data1Size);
    I_2 = ones(1, data2Size)*2;
    I_3 = ones(1, data3Size)*3;

    % losowanie zbioru uczacego
    for i = 1:train1Size
        sample = randi([1 train1Size+1-i]);
        X_t = [X_t X_u1(:,sample)];
        I = [I I_1(:,sample)];
        X_u1(:,sample) = [];
        I_1(:,sample) = [];
    end

    for i = 1:train2Size
        sample = randi([1 train2Size+1-i]);
        X_t = [X_t X_u2(:,sample)];
        I = [I I_2(:,sample)];
        X_u2(:,sample) = [];
        I_2(:,sample) = [];
    end

    for i = 1:train3Size
        sample = randi([1 train3Size+1-i]);
        X_t = [X_t X_u3(:,sample)];
        I = [I I_3(:,sample)];
        X_u3(:,sample) = [];
        I_3(:,sample) = [];
    end

    % zlozenie zbioru testowego
    X_s = [X_u1 X_u2 X_u3];
    I_s = [I_1 I_2 I_3];

    case 3
    %% ionosphere
    ion = load('ionosphere');
    fprintf('Zbior ionosphere, %g%% danych uczacych\n\n', d*100);

    % znalezienie minimow i maximow
    class = 2;
    data1Size = 225;
    data2Size = 126;
    dataSize = 351;
    train1Size = int16(data1Size * d);
    train2Size = int16(data2Size * d);
    trainSize = train1Size + train2Size;
    test1Size = data1Size - train1Size;
    test2Size = data2Size - train2Size;
    testSize = test1Size + test2Size;

    % wydobycie zbioru
    X_u1 = [];
    X_u2 = [];
    I_1 = [];
    I_2 = [];
    for i=1:dataSize
       if ion.Y{i} == 'g'
           X_u1 = [X_u1 ion.X(i,:)'];
           I_1 = [I_1 1];
       else
           X_u2 = [X_u2 ion.X(i,:)'];
           I_2 = [I_2 2];
       end
    end

    % przemieszanie zbioru
    X_u1 = X_u1(:,randperm(data1Size));
    X_u2 = X_u2(:,randperm(data2Size));

    % losowanie zbioru uczacego
    for i = 1:train1Size
        sample = randi([1 train1Size+1-i]);
        X_t = [X_t X_u1(:,sample)];
        I = [I I_1(:,sample)];
        X_u1(:,sample) = [];
        I_1(:,sample) = [];
    end

    for i = 1:train2Size
        sample = randi([1 train2Size+1-i]);
        X_t = [X_t X_u2(:,sample)];
        I = [I I_2(:,sample)];
        X_u2(:,sample) = [];
        I_2(:,sample) = [];
    end

    % zlozenie zbioru testowego
    X_s = [X_u1 X_u2];
    I_s = [I_1 I_2];

end

    neurons = 4;
    for opt={...
             {'ChoiD', 'Frank', 100, 0.55, 0.55}, ...
             {'ChoiD', 'Algebraic', 2, 0.09, 0.09},...
             {'ChoiD', 'Yager', 2, 0.17, 0.17},...
             {'ChoiD', 'Dombi', 2, 0.06, 0.06},...
             {'ChoiD', 'Hamacher', 10, 0.38, 0.38},...
             {'D', 'Frank', 100, 0.45, 0.45}, ...
             {'D', 'Algebraic', 2, 0.15, 0.15},...
             {'D', 'Yager', 2, 0.20, 0.20},...
             {'D', 'Dombi', 2, 0.92, 0.92},...
             {'D', 'Hamacher', 10, 0.57, 0.57},...
             {'JK', 'Frank', 100, 0.31, 0.31}, ...
             {'JK', 'Algebraic', 2, 0.21, 0.21},...
             {'JK', 'Yager', 2, 0.06, 0.06},...
             {'JK', 'Dombi', 2, 0.11, 0.11},...
             {'JK', 'Hamacher', 10, 0.32, 0.32}
            };

        display(opt{1});
        errors_t = zeros(1,10);
        clasif_t = zeros(1,10);
        for z=1:10
            display(z);
            net = FeedForwardNetwork(neurons,'Fuzzy','Fuzzy');
            net.FFNeuronOptions{1} = opt{1};
            net.FFNeuronOptions{2} = opt{1};
            net = configure(net, X_t', I');
            net = trainlm(net, X_t', I', 1e-4, 250, 1e9);

            err = 0;
            clas = 0;
            for j=1:testSize
                res = sim(net, X_s(:,j)');
                err = err + (res-I_s(j))^2;
                if abs(res - I_s(j)) > 0.1
                    clas = clas + 1;
                end
            end
            err = err / double(testSize);
            clas = clas / double(testSize);
            errors_t(z) = err;
            clasif_t(z) = err;
            display([err clas]);
        end

        dlmwrite(strcat('clasif/', opt{1}{1}, '_', opt{1}{2}, '_', ...
            int2str(neurons), '_', sets{ds}, '.txt'), [mean(err), std(err); mean(clas), std(clas)]);
    end

end