function [Z,E,time,elp] =  LADMAP(XX,omega,lambda,p)
tic;
X = XX.*omega;
A = X;
tol = 1e-5;
maxIter = 300;
[d, n] = size(X);
m = size(A,2);
rho = 1.2;
max_mu = 1e6;
mu = 1e-4;
atx = A'*X;
eta = norm(X, 2)^2 + 1;
elp=zeros(1,maxIter);
%% Initializing optimization variables
% intialize
J = zeros(m,n);
Z = zeros(m,n);
E = sparse(d,n);
Y1 = zeros(d,n);
Y2 = zeros(m,n);
%% Start main loop
iter = 0;
disp(['initial,rank=' num2str(rank(Z))]);
while iter<maxIter
    iter = iter + 1;
    
    %update J
    temp=Z -(1/eta)*X'*(X*Z+E-X+Y1/mu);
    [U,sigma,V] = svd(temp,'econ');
    sigma1 = diag(sigma);
    sigma2=tau_p(sigma1, 1/(mu*eta), p);
    sigma3=diag(sigma2);
    J=U*sigma3*V';
    %udpate Z
    Z = (A'*A+eye(m))\(atx-A'*E+J+(A'*Y1-Y2)/mu);
    
    %update E
    xmaz = X-A*Z;
    temp = xmaz+Y1/mu;
    E = max(0,temp - lambda/mu)+min(0,temp + lambda/mu);
    
    leq1 = xmaz-E;
    leq2 = Z-J;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    elp(iter) = stopC;
    
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM_l1=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end
time = toc;
disp(time);
end

function x_out = tau_p(y, lambda, p) 
    tau = (2 * lambda * (1 - p))^(1 / (2 - p)) + lambda*p * (2 * lambda * (1 - p))^((p - 1) / (2 - p));
    x_out = zeros(size(y));

 
    for i = 1:length(y)
        if abs(y(i)) <= tau
            x_out(i) = 0;
        else
            theta = solve_gp(y(i), lambda, p);
            x_out(i) = sign(y(i)) * theta;
        end
    end
end

function theta = solve_gp(y, lambda, p)
    g_p = @(theta) theta + lambda*p * theta^(p - 1) - abs(y);
    theta = fzero(g_p, abs(y)); 
end