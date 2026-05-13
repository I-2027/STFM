clear 
warning off all
data_dir='./dataset/USPS';
addpath(genpath(data_dir));
addpath('C:\Users\PC\Desktop\STFM\SC code\USPSÈý·Ö½â\USPS');
  load USPS_1000.mat

  fea= fea / 255;
  nClusters = 3:10;
  acci_3 = zeros(length(nClusters),1);
  t_3 = zeros(length(nClusters),1);
  nmi=zeros(length(nClusters),1);
 for i = 1:length(nClusters)
     nCluster =  nClusters(1,i);
      fea1 = fea(:,1:num1(nCluster));
      gnd1 = gnd(:,1:num1(nCluster));
      dim = nCluster * 13;
      %%PCA
      [ eigvector , eigvalue ] = PCA(fea1);
    data = eigvector(:,1:dim)'*fea1;
    for jj = 1 : size(data,2)
        data(:,jj) = data(:,jj)/norm(data(:,jj));  
    end
    omega=ones(size(data));
    [L_LRRsp,E,T_LRRsp,elp] =  LADMAP(data,omega, 0.01, 0.5);
     idx_3 = clu_ncut(L_LRRsp ,nCluster);
     acci_3(i) = compacc(idx_3,gnd1);
     t_3(i)=T_LRRsp;
     acc3=mean(acci_3);
     nmi(i)=NMI(idx_3,gnd1);
  end
  
  err_LRRsp_class = 1 - mean(acci_3,2);
  t_LRRsp_USPS = mean(t_3,2);
save LRRsp_e.mat err_LRRsp_class
save LRRsp_t_USPS.mat t_LRRsp_USPS
save elp.mat elp
save nmi_LADMAP.mat nmi