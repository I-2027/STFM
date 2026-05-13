function [L, S, time] = DWRPCA(M, r, lambda, mu, rho, maxIter, tol)
% DWRPCA: Dual-Weighted Robust Principal Component Analysis
% Input:
%   M: Input data matrix (m x n)
%   r: Target rank of low-rank matrix L
%   lambda: Regularization parameter for sparse term (default: 1/sqrt(max(m,n)))
%   mu: Initial penalty parameter (default: 1.25/norm(M,2))
%   rho: Multiplier for updating mu (default: 1.5)
%   maxIter: Maximum number of iterations (default: 1000)
%   tol: Tolerance for convergence (default: 1e-4)
% Output:
%   L: Low-rank matrix
%   S: Sparse matrix
tic
[m, n] = size(M);

% Set default parameters if not provided
if nargin < 3 || isempty(lambda)
    lambda = 1 / sqrt(max(m, n));
end
if nargin < 4 || isempty(mu)
    mu = 1.25 / norm(M, 2);
end
if nargin < 5 || isempty(rho)
    rho = 1.5;
end
if nargin < 6 || isempty(maxIter)
    maxIter = 800;
end
if nargin < 7 || isempty(tol)
    tol = 1e-5;
end

% Initialize variables
S = zeros(m, n);
L = zeros(m, n);
Y = mu * (M - L - S); % Lagrangian multiplier

% Initialize weights
w_S = ones(m, n); % Initial sparse weights
epsilon = 1e-6; % Small constant to avoid division by zero

% Precompute Frobenius norm of M for convergence check
normM = norm(M, 'fro');

% ADMM iteration
for iter = 1:maxIter
    % Store previous values for convergence check
    L_prev = L;
    S_prev = S;
    
    % Update S: S-subproblem
    X_S = M - L + (1/mu) * Y;
    threshold = lambda / mu * w_S;
    S = sign(X_S) .* max(abs(X_S) - threshold, 0);
    
    % Update L: L-subproblem
    X_L = M - S + (1/mu) * Y;
    
    % Compute truncated SVD (first r singular values/vectors)
    [U, Sigma, V] = svds(X_L, r);
    L = U * Sigma * V';
    
    % Update Y: Y-subproblem
    Y = Y + mu * (M - L - S);
    
    % Update sparse weights w_S
    w_S = 1 ./ (abs(S) + epsilon);
    
    % Update mu
    mu = min(mu * rho, 1e7);
    
    % Check convergence
    convergence = norm(M - L - S, 'fro') / normM;
    if convergence < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
    
    % Optional: Display progress every 100 iterations
    if mod(iter, 100) == 0
        fprintf('Iteration %d, convergence: %.6f\n', iter, convergence);
    end
end

if iter == maxIter
    fprintf('Maximum iterations reached. Convergence: %.6f\n', convergence);
end
time=toc;
disp(time)
end