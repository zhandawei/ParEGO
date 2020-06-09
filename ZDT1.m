function y = ZDT1(x, m)
% 0<xi<1
y(:,1)=x(:,1);
g=1+(9/(size(x,2)-1))*sum(x(:,2:end),2);
h=1-(y(:,1)./g).^0.5;
y(:,2)=g.*h;
end