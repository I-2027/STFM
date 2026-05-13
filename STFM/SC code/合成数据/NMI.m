function nmi = NMI(L1, L2)
    % 计算两个聚类结果的NMI值
    % 输入:
    %   L1, L2 - 两个聚类标签的向量，其中每个元素的值代表数据点所属的聚类
    % 输出:
    %   nmi - 归一化的互信息值

    % 计算每个聚类的大小
    n1 = length(L1);
    n2 = length(L2);
    n = n1; % 假设L1和L2长度相同
    
    % 如果任一聚类结果只有一个聚类，则NMI为0
    if length(unique(L1)) == 1 || length(unique(L2)) == 1
        nmi = 0;
        return;
    end

    % 构建联合分布矩阵
    [L1_sorted, idx1] = sort(L1);
    [L2_sorted, idx2] = sort(L2);
    L1 = L1_sorted;
    L2 = L2_sorted;
    u_L1=unique(L1);
    u_L2=unique(L2);
    N = zeros(length(u_L1), length(u_L2));
    
    for i = 1:length(unique(L1))
         for j = 1:length(unique(L2))
             N(i, j) = sum(((L1 == u_L1(i)) & (L2 == u_L2(j))));
         end
     end
    
    % 计算边缘分布
    P1 = sum(N, 2) / n;
    P2 = sum(N, 1) / n;
    
    % 计算互信息
    H1 = -sum(P1 .* log2(P1 + eps));
    H2 = -sum(P2 .* log2(P2 + eps));
    H12 = -sum(sum((N ./ n .* log2(((N ./ n) + eps)))));
    
    % 计算归一化互信息
    MI = H1 + H2 - H12;
    nmi = MI / sqrt(H1 * H2);
end