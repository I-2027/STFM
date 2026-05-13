function [M, E, Z, time,efg] = FGLRR(X,omega, lambda, p, q)
    
    tic;
    X=X.*omega;
    [n, m] = size(X);
    d=n;
    E = zeros(n, m);
    Z = zeros(m, m);
    L = eye(m, d);  
    R = eye(d, m);
    N_k = zeros(d, d);
    M = zeros(d, d);
    B_k = zeros(n, n);
    Y1 = zeros(n, m);
    Y2 = zeros(m, m);
    % Initialize 
    mu = 1e-5;
    rho = 1.1;
    mu_max=1e6;
    tol = 1e-4;
    maxIter = 200;
    eps=0.3;
    efg=zeros(1,maxIter);
    % Iterative optimization
    for iter = 1:maxIter
        
        % Update L
        D=Z+Y2/mu;
        [Q,~]=qr(D*R'); 
        L=Q(:,1:d);
        
        % Update R
        [Q2,~]=qr(D'*L);
        R1=Q2(:,1:d);        
        R=R1';
        
        % Update M
        for i = 1:d
            N_k(i, i) = (norm(M(:, i), 2)^2 + eps)^(p / 2 - 1);
        end
       
        temp = L' * (Z + Y2 / mu) * R';     
        M = temp /(( p * N_k)/mu + eye(size(N_k)));
        
        %update Z
        A = X - E + Y1 / mu;
        B = L * M * R - Y2 / mu;
        Z = (X' * X + eye(m))\(X' * A + B);
        
        %update E
        for i = 1:m
            B_k(i, i) = (norm(E(:, i), 2)^2 + eps)^(q / 2 - 1);
        end
         temp = (X - X * Z + Y1 / mu);
         E = temp *inv((q * B_k* lambda)/mu + eye(size(B_k))); 

        % Update Y1¡¢Y2
        Y1 = Y1 + mu * (X - X * Z - E);
        Y2 = Y2 + mu * (Z - L * M * R);
         mu = min(rho * mu, mu_max);
        efg(iter)=norm(X - X * Z - E, 'fro')/norm(X,'fro');
        % Convergence check
        if norm(X - X * Z - E, 'fro')/norm(X,'fro') < tol 
            fprintf('Converged in %d iterations.\n', iter);
            break;
        end
    end
    toc
    time=toc;
    disp(time)
end
