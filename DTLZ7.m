function y = DTLZ7(x,m)
% 0<xi<1   i=1,2...6
% x is a Vector

n=size(x,2);     %number of design variable
k=n-m+1;
g=1+(9/k)*sum(x(:,m:end),2);
y(:,1:m-1)=x(:,1:m-1);
h=m-sum((y(:,1:m-1)./(1+g(:,ones(1,m-1)))).*(1+sin(3*pi*y(:,1:m-1))),2);
y(:,m)=(1+g).*h;


end