function [idx, err] = clu_ncut(L,K)
% this routine groups the data X into K subspaces by NCut
% inputs:
%       L -- an N*N affinity matrix, N is the number of data points
%       K -- the number of subpaces (i.e., clusters)
L = (abs(L)+abs(L'))/2;

D = diag(sum(L,2).^(-1./2));
L = eye(size(L,1)) - D*L*D;
err = 0;
if sum(isinf(L),'all') || sum(isnan(L),'all')
    err = 1;
    idx = 0;
else
    [U,~,~] = svd(L);
    %% 
    V = U(:,end-K+1:end);
    idx = kmeans(V,K,'emptyaction','singleton','replicates',10,'display','off');
    idx = idx';
end

