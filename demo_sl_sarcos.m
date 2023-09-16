function  demo_sl_sarcos
% Demo: Riemannian natural gradient method for solving subspace learning
% problem on the Sarcos dataset

% -----------------------------------------------------------------------
% Reference:
%  J. Hu, R. Ao, A. M.-C. So, M. Yang, Z. Wen, 
%  Riemannian Natural Gradient Methods. SIAM Journal on Scientific Computing
%
%  Author: J. Hu, Z. Wen, R. Ao
%  Version 1.0 .... 2022/12


%% Real data or synthetic data
trainTestRatio = 0.8; %% between 0 and 1, larger implies larger training data
seed = 42; % setting randomization seed for consistent output
dataset = 'sarcos'; % values permitted: 'sarcos', 'school', 'parkinson'.

rng(seed);
% Details of datasets present in the file generateTrainTestSplit.m
[data1,data1_test, Utruth] = generateTrainTestSplit(dataset,trainTestRatio);

% Train data
problem.data = data1;

% Test data
problem.data_test = data1_test;
T = length(problem.data);
T2 = length(problem.data_test);

for t = 1 : T % Number of tasks for training
    problem.data(t).XtX = problem.data(t).X'*problem.data(t).X;
    problem.data(t).Xty = problem.data(t).X'*problem.data(t).y;
end

for t = 1 : T2 % Number of tasks for testing
    problem.data_test(t).XtX = problem.data_test(t).X'*problem.data_test(t).X;
    problem.data_test(t).Xty = problem.data_test(t).X'*problem.data_test(t).y;
end

%% Proposed

% Mandatory information.
problem.r = 6; % Rank
problem.m = size(problem.data(1).X, 2); % Number of features
problem.str = 'sl';

problem.M = grassmannfactory(problem.m, problem.r);

problem.cost = @func_value;

maxepoch = 1000;
tolgradnorm = 1e-6;
problem.egrad = @compute_grad;
problem.hess = @ehess;
Uinit = problem.M.rand();
clear options;
options.maxiter = maxepoch;
lambda = 1e-4;
r = problem.r;
m = problem.m;
options.tolgradnorm = tolgradnorm;

maxepoch = 90;
batchsize = 1;
inner_repeat = 1;
problem.ncostterms = T;
dir = "./results/sl/"+dataset+string(maxepoch)+"epoch";
mkdir  (dir);

problem.partialegrad = @partialegrad;
problem.egradw = @compute_gradw;
problem.partialegradw = @partialegradw;




% RNGD
options.verbosity = 1;
options.batchsize = 1;
options.update_type='rngd';
options.maxepoch = maxepoch / (1 + 1 * inner_repeat);
options.tolgradnorm = tolgradnorm;
options.rngd_type = 1;
options.stepsize_type = 'fix';
options.stepsize = 1; % 3e0 for random
options.boost = 0;
options.rngd_type = 1; % effective only for R-SVRG variants
options.transport = 'ret_vector';
options.maxinneriter = inner_repeat * T;
options.damping = 1e-6;
options.wupdate = 'minus';
options.statsfun = @sl_mystatsfun;
[~, ~, infos_rngd, options_rngd] = Riemannian_ngd(problem, Uinit, options);

%% plots
N = T;
num_grads_rngd = ceil((N + 1 * options_rngd.maxinneriter)/N)*((1:length([infos_rngd.cost])) - 1);
% Train MSE versus #grads/N
fs = 12;
figure;
set(gcf,'visible','off');
plot(num_grads_rngd, [infos_rngd.nmse], ':<','LineWidth',2,'Color', [0,0,0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs);
ylabel(ax1,'NMSE on training set','FontName','Arial','FontSize',fs);
legend('RNGD');
filename = string(dataset) + "train-epoch";
saveas(gcf, dir+'/'+filename+'.png','png');

% Train MSE versus #grads/N
fs = 12;
figure;
set(gcf,'visible','off');
plot([infos_rngd.time], [infos_rngd.nmse], ':<','LineWidth',2,'Color', [0,0,0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'time','FontName','Arial','FontSize',fs);
ylabel(ax1,'NMSE on training set','FontName','Arial','FontSize',fs);
legend('RNGD');
filename = string(dataset) + "train-time";
saveas(gcf, dir+'/'+filename+'.png','png');

% Test MSE versus #grads/N
fs = 12;
figure;
set(gcf,'visible','off');
plot(num_grads_rngd, [infos_rngd.nmse_test],':<','LineWidth',2,'Color', [0,0,0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs);
ylabel(ax1,'NMSE on test set','FontName','Arial','FontSize',fs);
legend('RNGD');
filename = string(dataset) + "test-epoch";
saveas(gcf, dir+'/'+filename+'.png','png');

% Test MSE versus #grads/N
fs = 12;
figure;
set(gcf,'visible','off');
plot([infos_rngd.time], [infos_rngd.nmse_test],':<','LineWidth',2,'Color', [0,0,0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel(ax1,'time','FontName','Arial','FontSize',fs);
ylabel(ax1,'NMSE on test set','FontName','Arial','FontSize',fs);
legend('RNGD');
filename = string(dataset) + "test-time";
saveas(gcf, dir+'/'+filename+'.png','png');


    function grad = compute_grad(U)
        w = compute_subprob(U, []);
        grad = zeros(m,r);
        for tt = 1:T
            wi = w(:,tt);
            grad = grad + problem.data(tt).XtX*(U*(wi*wi')) - problem.data(tt).Xty*wi';
        end
        grad = grad / T;
        %     grad = Grassmann(U,grad);
    end

    function [grad,w] = compute_gradw(U)
        w = compute_subprob(U, []);
        grad = zeros(m,r);
        for tt = 1:T
            wi = w(:,tt);
            grad = grad + problem.data(tt).XtX*(U*(wi*wi')) - problem.data(tt).Xty*wi';
        end
        grad = grad / T;
    end

    function g = partialegrad(U, idx_batchsize)
        g = zeros(m, r);
        m_batchsize = length(idx_batchsize);
        for ii = 1 : m_batchsize
            column = idx_batchsize(ii);
            w = compute_subprob(U, column);
            g = g + problem.data(column).XtX*(U*(w*w')) - problem.data(column).Xty*w';
        end
        g = g/m_batchsize;

    end

    function [g,W] = partialegradw(U, idx_batchsize)
        g = zeros(m, r);
        %         W = zeros(length(samples(idx_batchsize(1))), size(U, 2));
        m_batchsize = length(idx_batchsize);
        W = [];
        for ii = 1 : m_batchsize
            column = idx_batchsize(ii);
            w = compute_subprob(U, column);
            g = g + problem.data(column).XtX*(U*(w*w')) - problem.data(column).Xty*w';
            W = [W, w];
        end
        g = g/m_batchsize;
    end

    function stats = sl_mystatsfun(problem, U, stats)

        value = 0;
        nmse = 0;
        nmse_test = 0;
        for k = 1:T2
            w = ((U'*problem.data_test(k).XtX)*U + lambda *eye(r))\(U'*problem.data_test(k).Xty);
            tmp = problem.data_test(k).X*U*w-problem.data_test(k).y;
            value = value + 0.5*norm(tmp)^2;
            nmse_test = nmse_test + norm(tmp)^2/length(problem.data_test(k).y)/var(problem.data_test(k).y);
        end
        value = value / T2;
        nmse_test = nmse_test / T2;
        stats.cost_test = value;
        stats.nmse_test = nmse_test;

        for k = 1:T
            w = ((U'*problem.data(k).XtX)*U + lambda *eye(r))\(U'*problem.data(k).Xty);
            tmp = problem.data(k).X*U*w-problem.data(k).y;
            nmse = nmse + norm(tmp)^2/length(problem.data(k).y)/var(problem.data(k).y);
        end

        stats.nmse = nmse/T;
    end

    function hess = ehess(U,V)
        w = compute_subprob(U,[]);
        %         hess = zeros(m,r);
        %         for k = 1:T
        %             wi = w(:,k);
        %             hess = hess + problem.data(k).XtX*(V*(wi*wi'));
        %         end

        XtX = zeros(m,m);
        wtw = zeros(r,r);
        for k = 1:T
            wi = w(:,k);
            wtw = wtw + wi*wi';
            XtX = XtX + problem.data(k).XtX;
        end
        hess = XtX*V*wtw;
        hess = hess / T;
        hess = Grassmann(U,hess);
    end


    function value = func_value(U)
        w = compute_subprob(U,[]);
        value = 0;
        for k = 1:T
            value = value + 0.5*norm(problem.data(k).X*U*w(:,k)-problem.data(k).y, 'fro')^2;
        end
        value = value / T;
    end

    function w = compute_subprob(U, samples)
        if isempty(samples)
            samples = 1:T;
        end
        w = zeros(size(U, 2), length(samples));
        if length(samples) ==1
            w = ((U'*problem.data(samples).XtX)*U + lambda *eye(r))\(U'*problem.data(samples).Xty);
            return;
        end

        for tt = 1:length(samples)
            w(:,tt) = ((U'*problem.data(samples(tt)).XtX)*U + lambda *eye(r))\(U'*problem.data(samples(tt)).Xty);
        end
    end

    function projected = Grassmann(U,Y)
        projected = Y - U*(U'*Y);
    end
end