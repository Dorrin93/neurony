net = FeedForwardNetwork([3 4], 'Fuzzy', 'Fuzzy', 'Lin');

X = [1 2 3 4 5];
Y = [5 4 3 2 1];

net = configure(net, X, Y);