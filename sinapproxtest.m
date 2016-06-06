clc;
clear;
format long

%% parametry funkcji f(x) = sin(c1*x)*sin(c2*x)/2 + 0.5;
c1 = 20;
c2 = 7;

%% zbior uczacy
X_u = 0:0.02:1;
T_u = [];
for i=X_u
    T_u = [T_u sin(i*c1)*sin(i*c2)/2 + 0.5];
end


%% zbior testowy
X_t = 0:0.001:1;
n = size(X_t, 2);
T_t = zeros(1, n);

it = 1;
for i=X_t
    T_t(it) = sin(i*c1)*sin(i*c2)/2 + 0.5;
    it = it+1;
end

%% testowanie
X_u = X_u';
T_u = T_u';

for opt={%{'ChoiD', 'Frank', 100, 0.55, 0.55}, ...
        % {'ChoiD', 'Algebraic', 2, 0.09, 0.09},...
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
N_neu = [];
E_neu = [];
for i=2:2
    %display(i);
    fprintf('%g neurons\n', i);
    net = FeedForwardNetwork( [i ],'Fuzzy','Lin');
    net.FFNeuronOptions{1} = opt{1};
    %net.FFNeuronOptions{2} = {'ChoiD', 'Frank', 100, 0.55, 0.55};
    net = configure(net, X_u, T_u);
    %net = train_BMAM(net, X_u, T_u, 1e-4, 1e9);
    net = train_LM(net, X_u, T_u, 1e-4, 1000, 1e9);
    error = 0;
    T_n = zeros(1,n);
    for j = 1:n
        val = sim(net,X_t(j));
        error = error + (T_t(j) - val)^2;
        T_n(j) = val;
    end
    hold off;    
    fig = figure('visible','off');
    subplot(2,1,1);
    plot(X_t, T_t, 'b');
    hold on;
    plot(X_t, T_n, 'r');
    
    N_neu = [N_neu i];
    E_neu = [E_neu error/n];
    subplot(2,1,2);
    plot(N_neu, E_neu, '.-b');
    print(fig, strcat('plots/', opt{1}{1}, '_', opt{1}{2}, '_', int2str(i), '_lm'),'-dpng');
    close(fig);
end
dlmwrite(strcat('plots/', opt{1}{1}, '_', opt{1}{2}, '_lm_errors.txt'), E_neu);

end