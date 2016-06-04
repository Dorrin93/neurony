function y = y( x )
%Aproximated function

A=1;
B=6;
C=.5;
D=.6;
E=1;
%y = A * sin(B*x+C) * cos(D*x+E);
y=10/(x+exp(-2*x))+11;
%y=1/(x+0.01);
end

