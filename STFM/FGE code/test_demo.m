%% 加载所需路径
addpath('C:\Users\PC\Desktop\STFM\FGE code\utils');
addpath('C:\Users\PC\Desktop\STFM\FGE code\src');
addpath('C:\Users\PC\Desktop\STFM\FGE code\landsvd');
%% 加载数据
   input_avi = 'C:\Users\PC\Desktop\STFM\FGE code\dataset\demo.avi';

video = VideoReader(input_avi);
detector = vision.ForegroundDetector('NumGaussians', 2, 'NumTrainingFrames', 50);
 
num_frames = ceil(video.Duration * video.FrameRate);
ground_truth = false(video.Height, video.Width, num_frames);
 
for i = 1:num_frames
    frame = read(video, i);
    fg_mask = detector(frame);
    ground_truth(:, :, i) = fg_mask;
end
 
% 保存结果
save('ground_truth.mat', 'ground_truth');

%% 统一使用 VideoReader 读取视频
video = load_video_file(input_avi);
M = im2double(convert_video_to_2d(video));
[k,n,p] = deal(video.height, video.width, size(M,2));


%% 计算结果
omega = ones(size(M));

% 算法1：MTFsp
[~,~, E1, Z1, time1,~] = STFM(M, omega, 0.1, 0.1, 3);
L1 = M*Z1;
S1 = hard_threshold(E1);
% 算法2：PRPCA
opts.M = ones(size(M));
opts.nIters = 100;
[L2, Stil,stat] = robustPCA(M,2,0.001,opts);
time2=sum(stat.time);
S2 = hard_threshold(Stil);
% 算法3：DWRPCA
[L3, S3, time3] = DWRPCA(M, 2);
S3 = hard_threshold(S3);
[numr,numc] = size(M);
CM = randi([0 1],numr,numc); % simulated confidence map (binary matrix)
Omega = find(CM ~= 0);
[I, J] = ind2sub([numr numc],Omega);
[Z4, E4, ~, time5] = core_MAMR(M, 1.3, I, J);
L4=Z4; 
S4=hard_threshold(E4);
%% 评估指标计算
metrics = struct('MTFsp', [], 'FGLRR', [], 'LADMAP', [], 'IRFLLRR', []);

for algo = 1:4
    total_TP = 0; 
    total_FP = 0; 
    total_TN = 0; 
    total_FN = 0;

    for frame = 1:p
%         获取当前算法结果
        switch algo
            case 1
                S_frame = reshape(S1(:,frame), k, n);
            case 2
                S_frame = reshape(S2(:,frame), k, n);
            case 3
                S_frame = reshape(S3(:,frame), k, n);
            case 4
                S_frame = reshape(S4(:,frame), k, n);
        end
        
%         统一处理参数
        pred_mask = medfilt2(imbinarize(S_frame, 0.05), [5 5]);
        gt_mask = ground_truth(:,:,frame);

%         计算混淆矩阵
        TP = sum(gt_mask & pred_mask, 'all');
        FP = sum(~gt_mask & pred_mask, 'all');
        TN = sum(~gt_mask & ~pred_mask, 'all');
        FN = sum(gt_mask & ~pred_mask, 'all');
        
%         累加统计量
        total_TP = total_TP + TP;
        total_FP = total_FP + FP;
        total_TN = total_TN + TN;
        total_FN = total_FN + FN;
    end

%     计算全局指标
    recall = total_TP / (total_TP + total_FN + eps);
    precision = total_TP / (total_TP + total_FP + eps);
    f_measure = 2*(recall*precision)/(recall + precision + eps);
    pcc = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + eps);
    
%     存储结果
    switch algo
        case 1
            metrics.MTFsp = [recall, precision, f_measure, pcc];
        case 2
            metrics.PRPCA = [recall, precision, f_measure, pcc];  
        case 3
            metrics.DWRPCA = [recall, precision, f_measure, pcc];
        case 4
            metrics.MAMR = [recall, precision, f_measure, pcc];      
    end
end

%% 可视化对比（含ground truth）
frame_idx = 21; 
original_frame = reshape(M(:,frame_idx), k, n);
gt_frame = ground_truth(:,:,frame_idx);

% 算法结果
L1_frame = reshape(L1(:,frame_idx), k, n);
mask1 = medfilt2(imbinarize(reshape(S1(:,frame_idx),k,n), 0.05));

L2_frame = reshape(L2(:,frame_idx), k, n);
mask2 = medfilt2(imbinarize(reshape(S2(:,frame_idx),k,n), 0.05));

L3_frame = reshape(L3(:,frame_idx), k, n);
mask3 = medfilt2(imbinarize(reshape(S3(:,frame_idx),k,n), 0.05));

L4_frame = reshape(L4(:,frame_idx), k, n);
mask4 = medfilt2(imbinarize(reshape(S4(:,frame_idx),k,n), 0.05));

figure('Position', [100 100 1000 800], 'Name','ground truth对比');
subplot(1,6,1), imshow(original_frame), title('');
subplot(1,6,2), imshow(mask4), title('(a)');
subplot(1,6,3), imshow(mask2), title('(b)');
subplot(1,6,4), imshow(mask3), title('(c)');
subplot(1,6,5), imshow(mask1), title('(d)');
%% 控制台输出结果
fprintf('============= 四算法评估结果 =============\n');
fprintf('算法\t\tRecall\tPrecision\tF-measure\tPCC\n');
fprintf('MAMR\t%.4f\t%.4f\t\t%.4f\t\t%.4f\n', metrics.MAMR);
fprintf('PRPCA\t%.4f\t%.4f\t\t%.4f\t\t%.4f\n', metrics.PRPCA);
fprintf('DWRPCA\t%.4f\t%.4f\t\t%.4f\t\t%.4f\n', metrics.DWRPCA);
fprintf('MTFsp\t%.4f\t%.4f\t\t%.4f\t\t%.4f\n', metrics.MTFsp);


