function [Bi_Ir,Bt_Tr,Bi_Ie,Bt_Te,traintime,testtime] = main_JIMFH(I_tr, T_tr, I_te, T_te, bits, lambda, belta, gamma)

% Reference:
% Di Wang, Quan Wang, Lihuo He, Xinbo Gao and Yumin Tian. 
% Joint and Individual Matrix Factorization Hashing for Large-Scale Cross-Modal Retrieval. 
% Pattern Recognition, Volume 107, November 2020, 107479.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%

if ~exist('lambda','var')
    lambda = 0.5;
end
if ~exist('belta','var')
    belta = 0.001;
end
if ~exist('gamma','var')
    gamma = 0.0001;
end

traintime1 = cputime;
%% centralization
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

%% solve objective function
fprintf('start solving JIMFH...\n');
[Y, Y1, Y2, R] = solveJIMFH(I_tr', T_tr', lambda, belta, gamma, bits);
Y_total = [Y;Y2];

P1 = Y_total * I_tr / (I_tr' *I_tr + gamma * eye(size(I_tr,2)));
P2 = Y_total * T_tr / (T_tr' * T_tr + gamma * eye(size(T_tr,2)));

%% calculate hash codes
Yi_tr = sign((bsxfun(@minus, Y_total , mean(Y_total,2)))');
Yt_tr = sign((bsxfun(@minus, Y_total , mean(Y_total,2)))');
Yi_tr(Yi_tr<0) = 0;
Yt_tr(Yt_tr<0) = 0;
Bt_Tr = compactbit(Yt_tr);
Bi_Ir = compactbit(Yi_tr);
traintime2 = cputime;
traintime = traintime2 - traintime1;

testtime1 = cputime;
I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
Yi_te = sign((bsxfun(@minus,P1 * I_te' , mean(Y_total,2)))');
Yt_te = sign((bsxfun(@minus,P2 * T_te' , mean(Y_total,2)))');
Yi_te(Yi_te<0) = 0;
Yt_te(Yt_te<0) = 0;
Bt_Te = compactbit(Yt_te);
Bi_Ie = compactbit(Yi_te);
testtime2 = cputime;
testtime = testtime2 - testtime1;