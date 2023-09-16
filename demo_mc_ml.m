function  demo_mc_ml
% Demo: Riemannian natural gradient method for solving matrix completion
% problem on the MovieLens-1M

% -----------------------------------------------------------------------
% Reference:
%  J. Hu, R. Ao, A. M.-C. So, M. Yang, Z. Wen, 
%  Riemannian Natural Gradient Methods. SIAM Journal on Scientific Computing.
%
%  Author: J. Hu, Z. Wen, R. Ao
%  Version 1.0 .... 2022/12

%% set parameters
maxepoch = 30;
dir = "./results/mc/ml/"+string(maxepoch)+"epoch";
mkdir  (dir);
N = 3952;
d = 6040;
r = 5;
tolgradnorm = 1e-8;
batchsize = 50;
inner_repeat = 1;


%% generate dataset
[samples, samples_test, samples_valid, values, indicator, values_test, indicator_test, data_ls, data_test, ~] = generate_mc_data_ml();



%% set manifold
problem.M = grassmannfactory(d, r);
problem.ncostterms = N;

%% define problem
problem.str = 'mc';

% cost function
problem.cost = @mc_cost;
    function f = mc_cost(U)
        W = mylsqfit(U, samples);
        f = 0.5*norm(indicator.*(U*W') - values, 'fro')^2;
        f = f/N;
    end

% Euclidean  ient of the cost function
problem.egrad = @mc_egrad;
    function g = mc_egrad(U)
        W = mylsqfit(U, samples);
        g = (indicator.*(U*W') - values)*W;
        g = g/N;
    end

problem.egradw = @mc_egradw;
    function [g,W] = mc_egradw(U)
        W = mylsqfit(U, samples);
        g = (indicator.*(U*W') - values)*W;
        g = g/N;
        W = W';
    end

% Euclidean stochastic gradient of the cost function
problem.partialegrad = @mc_partialegrad;
    function g = mc_partialegrad(U, idx_batchsize)
        g = zeros(d, r);
        m_batchsize = length(idx_batchsize);
        for ii = 1 : m_batchsize
            colnum = idx_batchsize(ii);
            w = mylsqfit(U, samples(colnum));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
        end
        g = g/m_batchsize;

    end

% Euclidean stochastic gradient and Wu of the cost function
problem.partialegradw = @mc_partialegradw;
    function [g,W] = mc_partialegradw(U, idx_batchsize)
        g = zeros(d, r);
        %         W = zeros(length(samples(idx_batchsize(1))), size(U, 2));
        m_batchsize = length(idx_batchsize);
        W = [];
        for ii = 1 : m_batchsize
            colnum = idx_batchsize(ii);
            w = mylsqfit(U, samples(colnum));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
            W = [W;w];
        end
        g = g/m_batchsize;
        W = W';
    end

problem.hess = @hess;
    function h = hess(U,V)
        W = mylsqfit(U, samples);
        h = V*(W'*W);
        %         h = (pinv(W'*W) * V')';
        h = h/N;
    end

    function W = mylsqfit(U, currentsamples)
        W = zeros(length(currentsamples), size(U, 2));
        for ii = 1 : length(currentsamples)
            % Pull out the relevant indices and revealed entries for this column
            IDX = currentsamples(ii).indicator;
            values_Omega = currentsamples(ii).values;
            U_Omega = U(IDX,:);

            % Solve a simple least squares problem to populate U
            W(ii,:) = (U_Omega\values_Omega)';
        end
    end


%% run algorithms

% Initialize
Uinit = problem.M.rand();
% dir = "./fig";

% RNGD
clear options;
options.verbosity = 1;
options.batchsize = batchsize;
options.update_type='rngd';
options.maxepoch = maxepoch / (1 + inner_repeat);
options.tolgradnorm = tolgradnorm;
options.rngd_type = 1;
options.stepsize_type = 'fix';
options.stepsize = 2;
options.boost = 0;
options.rngd_type = 1; % effective only for R-SVRG variants
options.transport = 'ret_vector';
options.maxinneriter = inner_repeat * N;
options.wupdate = 'minus';
options.statsfun = @mc_mystatsfun;
[~, ~, infos_rngd, options_rngd] = Riemannian_ngd(problem, Uinit, options);


num_grads_rngd = ceil((N + options_rngd.maxinneriter)/N)*((1:length([infos_rngd.cost])) - 1);

%% plots
% Train MSE versus #grads/N
fs = 20;
figure;
set(gcf,'visible','off');
semilogy(num_grads_rngd, [infos_rngd.cost]  * 2 * N / data_ls.nentries,':<','LineWidth',2,'Color', [0,0,0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs);
ylabel(ax1,'MSE on training set','FontName','Arial','FontSize',fs);
legend('RNGD');
filename = "train-epoch";
saveas(gcf, dir+'/'+filename+'.png','png');

% Train MSE versus time
fs = 20;
figure;
set(gcf,'visible','off');
semilogy([infos_rngd.time], [infos_rngd.cost]  * 2 * N / data_ls.nentries,':<','LineWidth',2,'Color', [0,0,0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'time','FontName','Arial','FontSize',fs);
ylabel(ax1,'MSE on training set','FontName','Arial','FontSize',fs);
legend('RNGD');
filename = "train-time";
saveas(gcf, dir+'/'+filename+'.png','png');

% Test MSE versus #grads/N
fs = 20;
figure;
set(gcf,'visible','off');
semilogy(num_grads_rngd, [infos_rngd.cost_test]* 2 * N / data_test.nentries,':<','LineWidth',2,'Color', [0,0,0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs);
ylabel(ax1,'MSE on test set','FontName','Arial','FontSize',fs);
legend('RNGD');

filename = "test-epoch";
saveas(gcf, dir+'/'+filename+'.png','png');

%     Test MSE verus time
fs = 20;
figure;
set(gcf,'visible','off');
semilogy([infos_rngd.time], [infos_rngd.cost_test]* 2 * N / data_test.nentries,':<','LineWidth',2,'Color', [0,0,0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'time','FontName','Arial','FontSize',fs);
ylabel(ax1,'MSE on test set','FontName','Arial','FontSize',fs);
legend('RNGD');

filename = "test-time";
saveas(gcf, dir+'/'+filename+'.png','png');

    function [samples, samples_test, samples_valid, values, indicator, values_test, indicator_test, data_ls, data_test, data_valid] = generate_mc_data_ml()

        load('./dataset/ml/ml_mat.mat'); % Original size [6040,3952]


        samples_valid = [];
        data_valid = [];


        %% ------ [start] BM code from Scaled SGD -----
        %% Randomly select nu rows and creat the data structure
        nu = d; % Number of users selected


        p = randperm(size(A,1), nu);
        A = A(p, :); % Matrix of size nu-by-100
        Avec = A(:);

        Avecindices = 1:length(Avec);
        Avecindices = Avecindices';
        i = ones(length(Avec),1);
        i(Avec == 0) = 0;
        Avecindices_final = Avecindices(logical(i));
        [I, J] = ind2sub([size(A,1)  3952],Avecindices_final);

        Avecsfinall = Avec(logical(i));


        [Isort, indI] = sort(I,'ascend');


        data_real.rows = Isort;
        data_real.cols = J(indI);
        data_real.entries = Avecsfinall(indI);
        data_real.nentries = length(data_real.entries);

        % Test data: two ratings per user
        [~,IA,~] = unique(Isort,'stable');
        data_ts_ind = [];
        for ii = 1 : length(IA)
            if ii < length(IA)
                inneridx = randperm(IA(ii+1) - IA(ii), 10);
            else
                inneridx = randperm(length(data_real.entries) +1 - IA(ii), 10);
            end
            data_ts_ind = [data_ts_ind; IA(ii) + inneridx' - 1];
        end


        data_test.rows = data_real.rows(data_ts_ind);
        data_test.cols = data_real.cols(data_ts_ind);
        data_test.entries = data_real.entries(data_ts_ind);
        data_test.nentries = length(data_test.rows);


        % Train data
        data_ls = data_real;
        data_ls.rows(data_ts_ind) = [];
        data_ls.cols(data_ts_ind) = [];
        data_ls.entries(data_ts_ind) = [];
        data_ls.nentries = length(data_ls.rows);


        % Permute train data
        random_order = randperm(length(data_ls.rows));
        data_ls.rows = data_ls.rows(random_order);
        data_ls.cols = data_ls.cols(random_order);
        data_ls.entries = data_ls.entries(random_order);


        % Dimensions and options
        n = size(A, 1);
        m = size(A, 2);

        %% ------ [end] BM code from Scaled SGD -----



        %% for train set
        values = sparse(data_ls.rows, data_ls.cols, data_ls.entries, n, m);
        indicator = sparse(data_ls.rows, data_ls.cols, 1, n, m);

        % Creat the cells
        samples(m).colnumber = []; % Preallocate memory.
        for k = 1 : m
            % Pull out the relevant indices and revealed entries for this column
            idx = find(indicator(:, k)); % find known row indices
            values_col = values(idx, k); % the non-zero entries of the column

            samples(k).indicator = idx;
            samples(k).values = values_col;
            samples(k).colnumber = k;
        end


        %% for test set
        values_test = sparse(data_test.rows, data_test.cols, data_test.entries, n, m);
        indicator_test = sparse(data_test.rows, data_test.cols, 1, n, m);

        samples_test(m).colnumber = [];
        for k = 1 : m
            % Pull out the relevant indices and revealed entries for this column
            idx = find(indicator_test(:, k)); % find known row indices
            values_col = values_test(idx, k); % the non-zero entries of the column

            samples_test(k).indicator = idx;
            samples_test(k).values = values_col;
            samples_test(k).colnumber = k;
        end

        d = n;
        N = m;
    end


    function stats = mc_mystatsfun(problem, U, stats)

        W = mylsqfit(U, samples_test);
        f_test = 0.5*norm(indicator_test.*(U*W') - values_test, 'fro')^2;
        f_test = f_test/N;
        stats.cost_test = f_test;

        if ~isempty(samples_valid)
            W = mylsqfit(U, samples_valid);
            f_valid = 0.5*norm(indicator_valid.*(U*W') - values_valid, 'fro')^2;
            f_valid = f_valid/N;
            stats.cost_valid = f_valid;
        end

    end
end
