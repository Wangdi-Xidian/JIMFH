% Reference:
% Di Wang, Quan Wang, Lihuo He, Xinbo Gao and Yumin Tian. 
% Joint and Individual Matrix Factorization Hashing for Large-Scale Cross-Modal Retrieval. 
% Pattern Recognition, Volume 107, November 2020, 107479.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
clc;clear 
load mirflickr25k.mat
%% Calculate the groundtruth
GT = L_te*L_tr';
WtrueTestTraining = zeros(size(L_te,1),size(L_tr,1));
WtrueTestTraining(GT>0)=1;
%% Parameter setting
bit = 32; 
%% Learn JIMFH
[B_I,B_T,tB_I,tB_T] = main_JIMFH(I_tr, T_tr, I_te, T_te, bit);
%% Compute mAP
Dhamm = hammingDist(tB_I, B_T)';    
[~, HammingRank]=sort(Dhamm,1);
mapIT = map_rank(L_tr,L_te,HammingRank); 
Dhamm = hammingDist(tB_T, B_I)';    
[~, HammingRank]=sort(Dhamm,1);
mapTI = map_rank(L_tr,L_te,HammingRank); 
map = [mapIT(100),mapTI(100)]
