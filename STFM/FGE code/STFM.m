function [L,M, E, Z, time,Y1] = MTFsp_test_e(X,omega, lambda, p, q)
    tic;
    X=X.*omega;
    [n, m] = size(X);
    d = 5;%hall
    E = zeros(n, m);
    Z = zeros(m,m);
    L = eye(m, d); 
    M = zeros(d, d);
    B_k = zeros(m, m);
    M_old = M;
    
    % Initialize
    %hall
    Y1 = zeros(n, m);
    mu = 1e-1;
    rho = 1.38;
    mu_max = 3e3;
    tol = 1e-5;
    maxIter = 50;
    eps = 0.3;
    eta = norm(X, 2)^2 + 1;
    maxk = 1;
    gamma = 0.1;
    tau = 1e-2;
    rho2 = 1.38;
    tau_max = 1e4;
    %escalator
    
%   rho = 1.17;
%   rho = 1.40;
    
    % Iterative optimization
   for iter=1:maxIter
        for k=1: maxk
        % Update L
        D=Z;
        [Q,~]=qr(D*L); 
        L=Q(:,1:d);
        
        % Update M
        grident_M=mu * (L' * (L * M_old * L' - Z) * L);
        M_temp = M_old - (1 / (mu + gamma)) * grident_M;
        [U,sigma,V] = svd(M_temp,'econ');
        sigma1 = diag(sigma);
        sigma2 = tau_p(sigma1, 1/(mu*eta), p);%mu*eta
        sigma3 = diag(sigma2);
        M=U*sigma3*V';
        M_old=M;
        end
        

        %update Z
        A = X - E + Y1 / mu;
        B = L * M * L' ;
        Z = (mu*(X' * X) + tau*eye(m))\(mu*(X' * A) + tau*B);
%          Z(Z < 0) = 0;
        
        %update E
        for i = 1:m
            B_k(i, i) = (norm(E(:, i), 2)^2 + eps)^(q / 2 - 1);
        end
        temp = (X - X * Z + Y1 / mu);
        E = temp / ((q * B_k* lambda)/mu + eye(size(B_k))); 

        % Update Y1
        Y1 = Y1 + mu * (X - X * Z - E);
        mu = min(rho * mu, mu_max);
        tau = min(rho2 * tau, tau_max);
        leq1 = Z - L * M * L';
        leq2 = X - X*Z - E;
        stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
        % Convergence check
        if stopC < tol
            fprintf('收敛于第 %d 次迭代，目标值变化：%e\n', iter);
            break;
        end
   end
    toc
    time=toc;
    disp(time)
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