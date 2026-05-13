function [data, omega] = generate_data(nn,sr,spr)
% sr 可观测数据的比例
% spr 矩阵受脉冲噪音污染的比例
%n = 20;
n = nn / 5;
d = 4; 
D = 150;
[U,~,~] = svd(rand(D));
cids = [];
U1 = U(:,1:d); % 基矩阵 D X d
X1 = U1*rand(d,n); % 子空间 D X n
cids = [cids,ones(1,n)];

R = orth(rand(D)); % 随机旋转矩阵
U2 = R*U1;
X2 = U2*rand(d,n);
cids = [cids,2*ones(1,n)];

U3 = R*U2;
X3 = U3*rand(d,n);
cids = [cids,3*ones(1,n)];

U4 = R*U3;
X4 = U4*rand(d,n);
cids = [cids,4*ones(1,n)];

U5 = R*U4;
X5 = U5*rand(d,n);
cids = [cids,5*ones(1,n)];

X = [X1,X2,X3,X4,X5]; % D X dn 150 X 100 每一列代表一个样本的观测值
data.X = X;
data.cids = cids;
nr = 5; % 重复次数
data.Xs = cell(1,nr); 
[mX,nX] = size(X);
% 加高斯白噪声
gn = sqrt(0.01) * randn(mX,nX); % 均值为0，方差为0.01 的高斯分布的噪声
for j=1:nr
    inds = rand(1,nX)<=0.2; % 选取加入噪声的列
    data.Xs{1,j} = X;
    data.Xs{1,j}(:,inds) = X(:,inds) + gn(:,inds);
end

% 加脉冲噪音
pn = -1 + 2 * rand(mX,nX); % [-1,1]之间均匀分布的脉冲噪音
for j=1:nr
    inds = rand(1,nX)<=spr; % 选取加入噪声的列
    data.Xs{1,j}(:,inds) = X(:,inds) + pn(:,inds);
end

omega = repmat((rand(mX,1) <= sr),1,nX);


