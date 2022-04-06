% Demo for Multiview Subspace Clustering

clear
close all

rng('default') % For reproducibility

addpath(genpath('utils/'))
addpath(genpath('algs/'));

data_path = 'data/';

num_runs = 20; % number of runs

% Algorithm Settings
flag_HL_L21_TLD = 1;

Data_list = {
    'uci_digit.mat',
    'bbcsport_2view.mat',
    'yale.mat',
    'yaleB.mat',
    'ORL.mat',
    'NH.mat',
    'scene15.mat',
    'MIT.mat',
    'COIL20MV.mat',
    'Caltech101.mat',
};
Data_name = {
    'UCI digits',
    'BBCSport',
    'Yale',
    'Extended YaleB',
    'ORL',
    'Notting Hill',
    'Scene-15',
    'MITIndoor-67',
    'Coil-20',
    'Caltech101',
};
Data_views = {
    3,
    2,
    3,
    3,
    3,
    3,
    3,
    4,
    3,
    4,
};


% test_list = 1:10;
test_list = 3:3;

for t = test_list
    clear X
    %% Loading data
    fprintf('Testing %s...\n', Data_name{t}) 
    load(fullfile(data_path, Data_list{t}));

    
    for k=1:Data_views{t}
        eval(sprintf('X{%d} = double(X%d);', k, k));
    end

    cls_num = length(unique(gt));
    K = length(X);


    %% record 
    alg_name = {}; 
    alg_cpu = {};
    alg_C = {};     % clustering results
    alg_S = {};     % affinity matrices
    alg_out = {};

    alg_NMI = {};
    alg_ACC = {};
    alg_AR = {};
    alg_fscore = {};   
    alg_precision = {};
    alg_recall = {};

    alg_cnt = 1;

    %% Algs Running
    if flag_HL_L21_TLD
        Y = X;
        for iv=1:K
            [Y{iv}]=NormalizeData(X{iv});
        end

        opts = [];
        opts.maxIter = 200;
        opts.mul_rate = 1.2;  
        opts.flag_debug = 0;
        opts.nb_num = 8;

        best_params_list = {
            [0.01   0.5],
            [0.05   0.9],
            [0.1    0.1],
            [0.006  0.1],
            [0.1    1],
            [0.006	0.5], 
            [0.006	0.9],
            [0.006	0.1],
            [0.001  0.1],
            [0.002	1],
        };
        param = best_params_list{t};
        opts.lambda1 = param(1); 
        opts.lambda2 = param(2);

        for kk = 1:num_runs
            time_start = tic;

            [C_HL_L21_TLD, S_HL_L21_TLD, Out_HL_L21_TLD] = HL_L21_TLD(Y, cls_num, gt, opts);

            alg_name{alg_cnt} = 'HL_L12_TLD';
            alg_cpu{alg_cnt}(kk) = toc(time_start);
            alg_NMI{alg_cnt}(kk) = Out_HL_L21_TLD.NMI;
            alg_AR{alg_cnt}(kk) = Out_HL_L21_TLD.AR;
            alg_ACC{alg_cnt}(kk) = Out_HL_L21_TLD.ACC;
            alg_recall{alg_cnt}(kk) = Out_HL_L21_TLD.recall;
            alg_precision{alg_cnt}(kk) = Out_HL_L21_TLD.precision;
            alg_fscore{alg_cnt}(kk) = Out_HL_L21_TLD.fscore; 
            alg_C{alg_cnt}{kk} = C_HL_L21_TLD;
            alg_S{alg_cnt}{kk} = S_HL_L21_TLD;
            alg_Out{alg_cnt}{kk} = Out_HL_L21_TLD;
        end
        alg_cnt = alg_cnt + 1;
    end

    %% result table
    flag_report = 1;
    if flag_report
        fprintf('%6s\t%12s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\n',...\
            'Stats', 'Algs', 'CPU', 'NMI', 'AR', 'ACC', 'Recall', 'Pre', 'F-Score');
        for j = 1:alg_cnt-1
            fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
                'Mean', alg_name{j},mean(alg_cpu{j}),mean(alg_NMI{j}),mean(alg_AR{j}),...\
                mean(alg_ACC{j}),mean(alg_recall{j}),mean(alg_precision{j}),mean(alg_fscore{j}));
        end
        for j = 1:alg_cnt-1
            fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
                'Std', alg_name{j},std(alg_cpu{j}),std(alg_NMI{j}),std(alg_AR{j}),...\
                std(alg_ACC{j}),std(alg_recall{j}),std(alg_precision{j}),std(alg_fscore{j}));
        end
    end

end
