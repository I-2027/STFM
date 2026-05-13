clear 
warning off all
data_dir='./dataset/USPS';
addpath(genpath(data_dir));
addpath('C:\Users\PC\Desktop\STFM\SC code\USPSÈý·Ö½â\USPS');
  load USPS_1000.mat

  fea= fea / 255;
  nClusters = 3:10;
  acci_1 = zeros(length(nClusters),1);
  t_1 = zeros(length(nClusters),1);
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
    [Z,L,~,~,~,time] = solve_irfllrr(data ,omega, 35 ,0.3 ,0.92 , 4); 
    idx_1 = clu_ncut(Z ,nCluster);
     acci_1(i) = compacc(idx_1,gnd1);
     t_1(i)=time;
     acc=mean(acci_1);
     nmi(i)=NMI(idx_1,gnd1);
  end
  err_irfllrr_class = 1 - mean(acci_1,2);
  t_irfllrr_USPS = mean(t_1,2);
  save tfllrr_e.mat err_irfllrr_class
  save tfllrr_t_USPS.mat t_irfllrr_USPS
  save nmi_IRFLLRR.mat nmi