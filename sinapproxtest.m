clc;
clear;

%% parametry funkcji f(x) = A*sin(x*B + C)*cos(x*D + E);
A = 5;
B = 2;
C = 1;
D = 3.5;
E = 0.5;

%% zbior uczacy
X_u = 0:0.1:5;
T_u = [];
for i=0:0.1:5
    T_u = [T_u A*sin(i*B+C)*cos(i*D+E)];
end

%% zbior testowy
n = 1001;
X_t = 0:0.005:5;
T_t = zeros(1, n);

it = 1;
for i=0:0.005:5
    T_t(it) = A*sin(i*B+C)*cos(i*D+E);
    it = it+1;
end

%% testowanie
X_u = X_u';
T_u = T_u';

N_neu = [];
E_neu = [];
for i=15:17
    display(i);
    net = FeedForwardNetwork(i, 'Tansig', 'Lin');
    net = configure(net, X_u, T_u);
    net = train_LM(net, X_u, T_u, 1e-5, 1000, 1e9);
    error = 0;
    T_n = zeros(1,n);
    for j = 1:1001
        val = sim(net,X_t(j));
        error = error + abs(T_t(j) - val);
        T_n(j) = val;
    end
    
    subplot(2,1,1);
    hold off;
    plot(X_t, T_t, 'b');
    hold on;
    plot(X_t, T_n, 'r');
    
    N_neu = [N_neu i];
    E_neu = [E_neu error/1001];
    subplot(2,1,2);
    plot(N_neu, E_neu, '.-b');
    
    uiwait
end