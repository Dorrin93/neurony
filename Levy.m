function y = Levy(n)
c = .5;
u = zeros(1,n);

%y = [y (c/(1/cdf('norm',1-i/2 ,0,1)).^2 + u)];
y = (c./(norminv(ones(1,n)-rand(1,n)./2,0,1).^2) + u);

%y = c/(norminv(1-rand()/2,0,1)^2) + u;
%y = rand(1,n);
%y = y./(sum(y.^2)^.5).*l;

end
