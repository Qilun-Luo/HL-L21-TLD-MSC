function [C, S, Out] = HL_L21_TLD(X, cls_num, gt, opts)

%% Note: Multiview Subspace Clustering with HL-L21-TLD Model
% Input:
%   X:          data features
%   cls_num:    number of clusters
%   gt:         groun truth clusters
%   opts:       optional parameters
%               - maxIter: max iteration
%               - nb_num: number of neighbors
%               - lambda1,lambda2, etc.:  hyper-parameter
%               - tau, xi1, xi2, etc.: penalty parameter
%               - epsilon: stopping tolerance
% Outout:
%   C:          clusetering results
%   S:          affinity matrix
%   Out:        other output information, e.g. metrics, history

%% Parameter settings
N = size(X{1}, 2);
K = length(X); % number of views

% Default
maxIter = 200;
epsilon = 1e-7;
mul_rate = 1.2; 

lambda1 = 0.1; 
lambda2 = 0.1; 
gamma1 = 0.1; 
gamma2 = 0.1;

tau = 1e-5;
xi1 = 1e-5;        
xi2 = 1e-5; 
rho1 = 1e-8;        
rho2 = 1e-8; 

max_tau = 1e10;
max_xi1 = 1e10; 
max_xi2 = 1e10;
max_rho1 = 1e10; 
max_rho2 = 1e10; 

flag_debug = 0;
flag_obj = 0;
nb_num = 3;
mode = 3;

if ~exist('opts', 'var')
    opts = [];
end  
if  isfield(opts, 'maxIter');       maxIter = opts.maxIter;         end
if  isfield(opts, 'epsilon');       epsilon = opts.epsilon;         end
if  isfield(opts, 'lambda1');       lambda1 = opts.lambda1;         end
if  isfield(opts, 'lambda2');       lambda2 = opts.lambda2;         end
if  isfield(opts, 'gamma1');        gamma1 = opts.gamma1;           end
if  isfield(opts, 'gamma2');        gamma2 = opts.gamma2;           end
if  isfield(opts, 'tau');           tau = opts.tau;                 end
if  isfield(opts, 'max_tau');       max_tau = opts.max_tau;         end
if  isfield(opts, 'xi1');           xi1 = opts.xi1;                 end
if  isfield(opts, 'max_xi1');       max_xi1 = opts.max_xi1;         end
if  isfield(opts, 'xi2');           xi2 = opts.xi2;                 end
if  isfield(opts, 'max_xi2');       max_xi2 = opts.max_xi2;         end
if  isfield(opts, 'rho1');          rho1 = opts.rho1;               end
if  isfield(opts, 'max_rho1');      max_rho1 = opts.max_rho1;       end
if  isfield(opts, 'rho2');          rho2 = opts.rho2;               end
if  isfield(opts, 'max_rho2');      max_rho2 = opts.max_rho2;       end
if  isfield(opts, 'mul_rate');      mul_rate = opts.mul_rate;       end
if  isfield(opts, 'flag_debug');    flag_debug = opts.flag_debug;   end
if  isfield(opts, 'flag_obj');      flag_obj = opts.flag_obj;       end
if  isfield(opts, 'nb_num');        nb_num = opts.nb_num;           end
if  isfield(opts, 'mode');          mode = opts.mode;               end


%% Initialize...
Econcat = [];
for k=1:K
    Z{k} = zeros(N,N); 
    M{k} = zeros(N,N);
    G{k} = zeros(N,N);
    Q{k} = zeros(N,N);
    Y{k} = zeros(N,N);
    W{k} = zeros(N,N);
    U{k} = zeros(N,N);
    J{k} = zeros(N,N);
    H{k} = zeros(N,N);
    L{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N); 
    P{k} = zeros(size(X{k},1),N);   
    Econcat = [Econcat; E{k}];
end


iter = 0;
Isconverg = 0;
while(Isconverg == 0)
    if flag_debug
        fprintf('----processing iter %d--------\n', iter+1);
    end

    %% ------------------- Update L^k -------------------------------
    for k=1:K
        Weight{k} = my_constructW_PKN((abs(Z{k})+abs(Z{k}'))./2, nb_num);
        Diag_tmp = diag(sum(Weight{k}));
        L{k} = Diag_tmp - Weight{k};
    end

    %% ------------------- Update Z^k -------------------------------    
    for k=1:K
        tmp = (tau*G{k}-M{k})+X{k}'*(xi1*X{k}-xi1*E{k}+P{k})+(xi2*Q{k}-Y{k})+...\
            (rho1*W{k}-U{k})+(rho2*J{k}-H{k});
        Z{k}=(xi1*X{k}'*X{k}+(tau+xi2+rho1+rho2)*eye(N,N))\tmp;
    end

    %% ------------------- Update E^k -------------------------------
    C = [];
    for k=1:K    
        tmp = X{k}-X{k}*Z{k}+P{k}/xi1;
        C = [C; tmp];
    end
    [Econcat] = solve_l1l2(C, lambda1/xi1);
    start = 1;
    for k=1:K
        E{k} = Econcat(start:start + size(X{k},1) - 1,:);
        start = start + size(X{k},1);
    end
  
    %% ------------------- Update Q^k -------------------------------
    for k=1:K
        Q{k} = (xi2*Z{k} + Y{k})/(lambda2*(L{k}+L{k}') + xi2*eye(N,N));
    end

    %% ------------------- Update W^k -------------------------------
    for k=1:K
        W{k} = solve_l1l2(Z{k}+U{k}/rho1, 2*gamma1/rho1);
    end

    %% ------------------- Update J^k -------------------------------
    for k=1:K
        J{k} = update_J(Z{k}+H{k}/rho2, 2*gamma2/rho2);
    end

    %% ------------------- Update G ---------------------------------
    Z_tensor = cat(3, Z{:,:});
    M_tensor = cat(3, M{:,:});
    clear M
    [G_tensor, ~] = logDet_Shrink(Z_tensor + M_tensor/tau, 1/tau, mode); % Logdet

    %% ------------------- Record the Info --------------------------
    % MATLAB 2021a or later is required for function "pagesvd"
    if flag_obj
        Zs_tensor = shiftdim(Z_tensor, 1);
        Zf_tensor = fft(Zs_tensor, [], 3);
        [~, ZfS, ~] = pagesvd(Zf_tensor, "econ");
        Z_TLD_norm = 0;
        for ii=1:N
            Z_TLD_norm = Z_TLD_norm + sum(log(1+diag(ZfS(:,:,ii)).^2));
        end
        X_XZ_mat = [];
        HL_norm = 0;
        Z_21_norm = 0;
        Z_12_norm = 0;
        for k = 1:K
            X_XZ_mat = [X_XZ_mat; X{k}-X{k}*Z{k}];
            HL_norm = HL_norm + trace(Z{k}*(L{k}+L{k}')/2*Z{k}');
            Z_21_norm = Z_21_norm + sum(vecnorm(Z{k}));
            Z_12_norm = Z_12_norm + sum(vecnorm(Z{k}, 1).^2);
        end
        obj_total = Z_TLD_norm/N + lambda1*sum(vecnorm(X_XZ_mat)) ... 
            + lambda2*HL_norm + gamma1*Z_21_norm + gamma2*Z_12_norm;
        history.objval(iter+1) = obj_total;
    end

    %% ------------------- Update auxiliary variables ---------------
    M_tensor = M_tensor  + tau*(Z_tensor - G_tensor);
    for k=1:K
        P{k} = P{k} + xi1*(X{k}-X{k}*Z{k}-E{k});
        Y{k} = Y{k} + xi2*(Z{k}-Q{k});  
        U{k} = U{k} + rho1*(Z{k}-W{k});
        H{k} = H{k} + rho2*(Z{k}-J{k});
        G{k} = G_tensor(:,:,k);
        M{k} = M_tensor(:,:,k);
    end   
    
    %% ------------------- Converge check ---------------------------
    Isconverg = 1;
    history.norm_Z(iter+1)= max(cellfun(@(x,z,e) norm(x-x*z-e, inf), X, Z, E));
    if (history.norm_Z(iter+1)>epsilon)
        if flag_debug
            fprintf('    norm_Z     %7.10f    \n', history.norm_Z(iter+1));
        end
        Isconverg = 0;
    end 

    history.norm_Z_G(iter+1) = max(cellfun(@(z,g) norm(z-g, inf), Z, G));
    if (history.norm_Z_G(iter+1)>epsilon)      
        if flag_debug
            fprintf('    norm_Z_G   %7.10f    \n', history.norm_Z_G(iter+1));
        end
        Isconverg = 0;
    end    

    history.norm_Z_Q(iter+1) = max(cellfun(@(z,q) norm(z-q, inf), Z, Q));
    if (history.norm_Z_Q(iter+1)>epsilon)   
        if flag_debug
            fprintf('    norm_Z_Q   %7.10f    \n', history.norm_Z_Q(iter+1));
        end
        Isconverg = 0;
    end 

    history.norm_Z_W(iter+1) = max(cellfun(@(z,w) norm(z-w, inf), Z, W));
    if (history.norm_Z_W(iter+1)>epsilon)
        if flag_debug
            fprintf('    norm_Z_W   %7.10f    \n', history.norm_Z_W(iter+1));
        end
        Isconverg = 0;
    end  

    history.norm_Z_J(iter+1) = max(cellfun(@(z,j) norm(z-j, inf), Z, J));
    if (history.norm_Z_J(iter+1)>epsilon)  
        if flag_debug
            fprintf('    norm_Z_J   %7.10f    \n', history.norm_Z_J(iter+1));
        end
        Isconverg = 0;
    end
    
    if (iter>maxIter)
        Isconverg  = 1;
    end
    
    %% ------------------- Update penalty params --------------------
    tau = min(tau*mul_rate, max_tau);
    xi1 = min(xi1*mul_rate, max_xi1);
    xi2 = min(xi2*mul_rate, max_xi2);
    rho1 = min(rho1*mul_rate, max_rho1);
    rho2 = min(rho2*mul_rate, max_rho2);
       
    iter = iter + 1;
end

%% ---------------- Clustering --------------------------------------
S = 0;
for k=1:K
    S = S + abs(Z{k})+abs(Z{k}');
end
C = SpectralClustering(S,cls_num);

[~, nmi, ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,p,r] = compute_f(gt,C);
[AR,~,~,~]=RandIndex(gt,C);

%% ---------------- Record ------------------------------------------
Out.NMI = nmi;
Out.AR = AR;
Out.ACC = ACC;
Out.recall = r;
Out.precision = p;
Out.fscore = f;
Out.history = history;
Out.Z = Z;

end


