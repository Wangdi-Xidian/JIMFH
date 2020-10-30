function [Y, Y1, Y2, R, obj] = solveJIMFH( X1, X2, lambda, belta, gamma, bits )

% Reference:
% Di Wang, Quan Wang, Lihuo He, Xinbo Gao and Yumin Tian. 
% Joint and Individual Matrix Factorization Hashing for Large-Scale Cross-Modal Retrieval. 
% Pattern Recognition, Volume 107, November 2020, 107479.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%

%% random initialization
col = size(X1,2);
Ibits = 3*bits/4;
Sbits = bits/4;
Y = rand(Sbits, col);
Y1 = rand(Ibits, col);
Y2 = rand(Ibits, col); 
R = rand(Ibits, Ibits);
threshold = 0.001;
lastF = 99999999;
iter = 1;
maxIter = 100;
obj = zeros(maxIter, 1);

%% compute iteratively
while (true)
    % update R
    R = Y2 * Y1' / (Y1 * Y1' + (gamma/belta) * eye(Ibits));
    
	% update U1, U2, P1, P2
    U1 = X1 * Y' / (Y * Y' + (gamma/lambda) * eye(Sbits));
    U2 = X2 * Y' / (Y * Y' + (gamma/(1-lambda)) * eye(Sbits));
    P1 = X1 * Y1' / (Y1 * Y1' + (gamma/lambda) * eye(Ibits));
    P2 = X2 * Y2' / (Y2 * Y2' + (gamma/(1-lambda)) * eye(Ibits));
    
	% update Y    
    Y = (lambda * U1' * U1 + (1- lambda) * U2' * U2 + gamma * eye(Sbits)) \ (lambda * U1' * X1 + (1 - lambda) * U2' * X2);
    
    % update Y1 and Y2
    Y1 = (lambda * P1' * P1 + belta * R'* R + gamma * eye(Ibits)) \ (lambda * P1' * X1 + belta * R'* Y2);
    Y2 = ((1-lambda) * P2' * P2 + belta * eye(Ibits) + gamma * eye(Ibits)) \ ((1 - lambda) * P2' * X2  + belta * R * Y1);
        
    % compute objective function
    norm1 = lambda * norm(X1 - U1 * Y, 'fro') + (1 - lambda) * norm(X2 - U2 * Y, 'fro');
    norm2 = lambda * norm(X1 - P1 * Y1, 'fro') + (1 - lambda) * norm(X2 - P2 * Y2, 'fro');
    norm3 = belta * norm(Y2 - R * Y1, 'fro');
    norm4 = gamma * (norm(U1, 'fro') + norm(U2, 'fro') + norm(Y, 'fro') + norm(P1, 'fro') + norm(P2, 'fro') + norm(Y1, 'fro') + norm(Y2, 'fro') + norm(R, 'fro'));
    currentF= norm1 + norm2 + norm3 + norm4;
    obj(iter) = currentF;
    fprintf('\nobj at iteration %d: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for seperable matrix factorization: %.4f,\n reconstruction error for consistency: %.4f,\n regularization term: %.4f\n\n', iter, currentF, norm1, norm2, norm3, norm4);
    if abs(lastF - currentF) < threshold
        fprintf('algorithm converges...\n');
        fprintf('final obj: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for seperable matrix factorization: %.4f,\n reconstruction error for consistency: %.4f,\n regularization term: %.4f\n\n', currentF, norm1, norm2, norm3, norm4);
        return;
    end
    if iter>=maxIter
        return
    end
    iter = iter + 1;
    lastF = currentF;
end
return;
end

