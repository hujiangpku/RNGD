function [x, cost, info, options] = Riemannian_ngd(problem, x, options)
% The Riemannian RNGD algorithms.
%
% function [x, cost, info, options] = Riemannian_rngd(problem)
% function [x, cost, info, options] = Riemannian_rngd(problem, x)
% function [x, cost, info, options] = Riemannian_rngd(problem, x, options)
% function [x, cost, info, options] = Riemannian_rngd(problem, [], options)
%
% -----------------------------------------------------------------------
% Reference:
%  J. Hu, R. Ao, A. M.-C. So, M. Yang, Z. Wen, 
%  Riemannian Natural Gradient Methods.
%
%  Author: J. Hu, Z. Wen, R. Ao
%  Version 1.0 .... 2022/12

% Verify that the problem description is sufficient for the solver.
if ~canGetCost(problem)
    warning('manopt:getCost', ...
        'No cost provided. The algorithm will likely abort.');
end
if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
    % Note: we do not give a warning if an approximate Hessian is
    % explicitly given in the problem description, as in that case the user
    % seems to be aware of the issue.
    warning('manopt:getGradient:approx', ...
        ['No gradient provided. Using an FD approximation instead (slow).\n' ...
        'It may be necessary to increase options.tolgradnorm.\n' ...
        'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
    problem.approxgrad = approxgradientFD(problem);
end

if ~canGetPartialGradient(problem)
    warning('manopt:getPartialGradient', ...
        'No partial gradient provided. The algorithm will likely abort.');
end

% If no initial point x is given by the user, generate one at random.
if ~exist('x', 'var') || isempty(x)
    x = problem.M.rand();
end



% Total number of samples
N = problem.ncostterms;

% Set local defaults
localdefaults.maxepoch = 100;  % Maximum number of epochs.
localdefaults.maxinneriter = 5*N;  % Maximum number of sampling per epoch.
localdefaults.stepsize = 0.1;  % Initial stepsize guess.
localdefaults.stepsize_type = 'decay'; % Stepsize type. Other possibilities are 'fix' and 'hybrid'.
localdefaults.stepsize_lambda = 0.1; % lambda is a weighting factor while using stepsize_typ='decay'.
localdefaults.tolgradnorm = 1.0e-6; % Batch grad norm tolerance.
localdefaults.batchsize = 1;  % Batchsize.
localdefaults.verbosity = 0;  % Output verbosity. Other localdefaults are 1 and 2.
localdefaults.boost = false;   % True: do a normal SGD at the first epoch when SVRG.
localdefaults.update_type = 'rngd';   % Update type. Other possibility is 'sgd', which is the standard SGD.
localdefaults.store_innerinfo = false; % Store information at each update. High memory requirements. Only to be used for debugging.
localdefaults.rngd_type = 1;  % To implement both the localdefaults that are used to define x0.
localdefaults.transport = 'ret_vector';
localdefaults.cgiter = 100;

% Merge global and local defaults, then merge w/ user options, if any.
localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);



stepsize0 = options.stepsize;
batchsize = options.batchsize;


% Total number of batches
totalbatches = ceil(options.maxinneriter/batchsize);



% Create a store database and get a key for the current x
storedb = StoreDB(options.storedepth);
key = storedb.getNewKey();


% Compute objective-related quantities for x
%     [cost, grad] = getCostGrad(problem, x, storedb, key);
%     cost = my_getcost(problem, x, storedb, key);
[cost, grad, w] = mygetCostGradw(problem, x, storedb, key);
wwt0 = w*w';

if strcmp(problem.str, 'sl')
    s = size(x,1);
    I = eye(s);
    data = problem.data;
    AtA = zeros(s,s);
    for k = 1:N
        AtA = AtA + data(k).XtX;
    end
    % AtA = AtA/N;
end
    
gradnorm = problem.M.norm(x, grad);

% Save stats in a struct array info, and preallocate.
epoch = 0;
stats = savestats();
info(1) = stats;
info(min(10000, options.maxepoch+1)).epoch = [];
info(min(10000, options.maxepoch+1)).cost = [];
info(min(10000, options.maxepoch+1)).time = [];
info(min(10000, options.maxepoch+1)).gradnorm = [];

% Initialize innerinfo
iter = 0;
inneriter = 0;
cgiter = 100;
if options.store_innerinfo
    innerstats = saveinnerstats();
    innerinfo(1) = innerstats;
    info(1).innerinfo = innerinfo;
    innerinfo(min(10000, totalbatches+1)).inneriter = [];
end


if options.verbosity > 0
    fprintf('-------------------------------------------------------\n');
    fprintf('R-%s:  epoch\t               cost val\t    grad. norm\t stepsize\n', options.update_type);
    fprintf('R-%s:  %5d\t%+.16e\t%.8e\t%.8e\n', options.update_type, 0, cost, gradnorm,stepsize0);
    
    
    if options.verbosity > 1
        fprintf('             inneriter\t               cost val\t    grad. norm\n');
    end
end


x0 = x;
grad0 = grad;
%     wwtHt = @(Ht) wwt0*Ht;
%     d0 = myCG(wwtHt, grad0', 0.1, cgiter);
%     d0 = d0';
%     d0 = wwt0 \ grad0';
%     d0 = d0';
toggle = 0; % To check boosting.
% Main loop over epoch.
for epoch = 1 : options.maxepoch
    
    
    % Draw the samples with replacement.
    perm_idx = randi(N, 1, options.maxinneriter);
    
    
    % Check if boost is required for rngd
    if strcmp(options.update_type, 'rngd') && options.boost && epoch == 1
        options.update_type = 'sgd';
        toggle = 1;
    end
    
    if strcmp(options.update_type, 'rngd') && options.rngd_type == 2
        update_instance = randi(totalbatches, 1) - 1; % pick a number uniformly between 0 to m - 1.
        if update_instance == 0
            xsave = x0;
            gradsave = grad0;
        end
    end
    
    
    elapsed_time = 0;
    % Per epoch: main loop over samples.
    for inneriter = 1 : totalbatches
        
        % Set start time
        start_time = tic;
        
        % Pick a sample of size batchsize
        start_index = (inneriter - 1)* batchsize + 1;
        end_index = min(inneriter * batchsize, options.maxinneriter);
        idx_batchsize = perm_idx(start_index : end_index);
        
        % Compute the gradient on this batch.
        %             partialgrad = getPartialGradient(problem, x, idx_batchsize, storedb, key);
        [partialgrad, partialw] = mygetPartialGradientw(problem, x, idx_batchsize, storedb, key);
        % compute the Fisher matrix
        partialwwt = partialw*partialw';
        %             wwtHt = @(Ht) wwt*Ht;
        %             partiald = myCG(wwtHt, partialgrad', 0.1, cgiter)';
        
        % Update stepsize
        if strcmp(options.stepsize_type, 'decay')
            stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
            
        elseif strcmp(options.stepsize_type, 'fix')
            stepsize = stepsize0; % Fixed stepsize.
            
        elseif strcmp(options.stepsize_type, 'hybrid')
            %if epoch < 5 % Decay stepsize only for the initial few epochs.
            if epoch < 3 % Decay stepsize only for the initial few epochs.              % HK
                stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
            end
            
        else
            error(['Unknown options.stepsize_type. ' ...
                'Should be fix or decay.']);
        end
        
        
        % Update partialgrad
        if strcmp(options.update_type, 'rngd')
            
            if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')
                
                % Logarithm map
                logmapX0ToX = problem.M.log(x0, x);
                
                % Parallel translate from U0 to U.
                grad0_transported = problem.M.paratransp(x0, logmapX0ToX, grad0);
                %                     d0_transported = problem.M.paratransp(x0, logmapX0ToX, d0);
                
                
                % Caclculate partialgrad at x0
                %                     partialgrad0 = getPartialGradient(problem, x0, idx_batchsize, storedb, key);
                [partialgrad0, partialw0] = mygetPartialGradientw(problem, x0, idx_batchsize, storedb, key);
                partialwwt0 = partialw0*partialw0';
                %                     wwtHt = @(Ht) partialwwt0*Ht;
                %                     partiald0 = myCG(wwtHt, partialgrad0', 0.1, cgiter)';
                
                % Caclculate transported partialgrad from x0 to x
                partialgrad0_transported = problem.M.paratransp(x0, logmapX0ToX, partialgrad0);
                %                     partiald0_transported = problem.M.paratransp(x0, logmapX0ToX, partiald0);
            elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')
                
                % Logarithm map
                %logmapX0ToX = problem.M.log(x0, x);
                %VecX0ToX = logmapX0ToX;
                % Projection of ( x-x0 )
                VecX0ToX = problem.M.proj(x0, x-x0);
                % vector transport from x0 to x.
                grad0_transported = problem.M.transp_locking(x0, VecX0ToX, x, grad0);
                %                     d0_transported = problem.M.transp_locking(x0, VecX0ToX, x, d0);
                
                
                % Caclculate partialgrad at x0
                %                     partialgrad0 = getPartialGradient(problem, x0, idx_batchsize, storedb, key);
                [partialgrad0, partialw0] = mygetPartialGradientw(problem, x0, idx_batchsize, storedb, key);
                partialwwt0 = partialw0*partialw0';
                %                     wwtHt = @(Ht) partialwwt0*Ht;
                %                     partiald0 = myCG(wwtHt, partialgrad0', 0.1, cgiter)';
                
                % Caclculate transported partialgrad from x0 to x
                partialgrad0_transported = problem.M.transp_locking(x0, VecX0ToX, x, partialgrad0);
                %                     partiald0_transported = problem.M.transp_locking(x0, VecX0ToX, x, partiald0);
                
                
            else
                
                % Caclculate transported full batch gradient from x0 to x.
                grad0_transported = problem.M.transp(x0, x, grad0); % Vector transport.
                %                     d0_transported = problem.M.transp(x0, x, d0);
                
                % Caclculate partialgrad at x0
                [partialgrad0, partialw0] = mygetPartialGradientw(problem, x0, idx_batchsize, storedb, key);
                partialwwt0 = partialw0*partialw0';
                %                     wwtHt = @(Ht) partialwwt0*Ht;
                %                     partiald0 = myCG(wwtHt, partialgrad0', 0.1, cgiter)';
                
                
                % Caclculate transported partialgrad from x0 to x
                partialgrad0_transported = problem.M.transp(x0, x, partialgrad0); % Vector transport.
                %                     partiald0_transported = problem.M.transp(x0, x, partiald0);
            end
            
            
            
            % Update partialgrad to reduce variance by
            % taking a linear combination with old gradients.
            % We make the combination
            % partialgrad + grad0 - partialgrad0.
            % partialwwt + wwt - partialwwt0
            partialgrad = problem.M.lincomb(x, 1, grad0_transported, 1, partialgrad);
            partialgrad = problem.M.lincomb(x, 1, partialgrad, -1, partialgrad0_transported);
            
            if strcmp(options.wupdate, 'average') == 1
                    wwt = ((problem.ncostterms-1)*wwt0 / problem.ncostterms + partialwwt /problem.ncostterms)/problem.ncostterms;    
                elseif strcmp(options.wupdate, 'minus') == 1
                    wwt = (wwt0 + partialwwt - partialwwt0)/problem.ncostterms;
                else
                    wwt = partialwwt / problem.ncostterms;
            end
            
            if strcmp(problem.str, 'sl')
                BtB = AtA - x*(x'*AtA); BtB = BtB - (BtB*x)*x' + options.damping*I;
                partiald = BtB \ (partialgrad*pinv(wwt));
            else
                partiald = partialgrad*pinv(wwt);
            end
%             partiald = AtA*partialgrad*pinv(wwt)';
            %                 partiald = problem.M.lincomb(x, 1, d0_transported, 1, partiald);
            %                 partiald = problem.M.lincomb(x, 1, partiald, -1, partiald0_transported);
            
        elseif strcmp(options.update_type, 'sgd')
            % Do nothing
            
        else
            error(['Unknown options.update_type. ' ...
                'Should be rngd or sgd.']);
            
        end
        
        % Update x
        if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')
            %                 xnew =  problem.M.exp(x, partialgrad, -stepsize);
            xnew =  problem.M.exp(x, partiald, -stepsize);
            
        elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')
            %                 xnew =  problem.M.exp(x, partialgrad, -stepsize);
            xnew =  problem.M.exp(x, partiald, -stepsize);
            
        else
            %                 xnew =  problem.M.retr(x, partialgrad, -stepsize);
            xnew =  problem.M.retr(x, partiald, -stepsize);
        end
        newkey = storedb.getNewKey();
        
        % Elapsed time
        elapsed_time = elapsed_time + toc(start_time);
        
        iter = iter + 1; % Total number updates.
        
        if strcmp(options.update_type, 'rngd') && options.rngd_type == 2 && inneriter == update_instance
            xsave = xnew;
            gradsave = getGradient(problem, xnew);
        end
        
        if options.store_innerinfo
            newcost = problem.cost(xnew);
            newpartialgradnorm = problem.M.norm(xnew, partialgrad);
            cost = newcost;
            key = newkey;
            partialgradnorm = newpartialgradnorm;
            
            innerstats = saveinnerstats();
            innerinfo(inneriter) = innerstats;
            if options.verbosity > 1
                fprintf('R-%: %5d (%5d)\t%+.16e\t%.8e\t%.8e\n', options.update_type, inneriter, epoch, cost, partialgradnorm, stepsize);
            end
        end
        
        x = xnew;
        key = newkey;
    end
    
    % Calculate cost, grad, and gradnorm
    if strcmp(options.update_type, 'rngd') && options.rngd_type == 2
        x0 = xsave;
        grad0 = gradsave;
    else
        if strcmp(options.update_type, 'rngd')
            trngd = tic;
        end
        
        x0 = xnew;
        %             grad0 = getGradient(problem, xnew);
        [cost, grad0, w] = mygetCostGradw(problem, xnew, storedb, key);
        wwt0 = w*w';

        if strcmp(options.update_type, 'rngd')
            elapsed_time = elapsed_time + toc(trngd);
        end
        
    end
    [newcost, newgrad] = getCostGrad(problem, xnew,storedb, newkey);
    newgradnorm = problem.M.norm(xnew, newgrad);
    
    % Transfer iterate info
    x = xnew;
    cost = newcost;
    key = newkey;
    gradnorm = newgradnorm;
    
    % Log statistics for freshly executed iteration
    stats = savestats();
    
    if options.store_innerinfo
        stats.innerinfo = innerinfo;
    end
    info(epoch+1)= stats;
    if options.store_innerinfo
        info(epoch+1).innerinfo = innerinfo;
    end
    
    % Reset if boosting used already.
    if toggle == 1
        options.update_type = 'rngd';
    end
    
    % Print output
    if options.verbosity > 0
        fprintf('R-%s:  %5d\t%+.16e\t%.8e\t%.8e\n',options.update_type, epoch, cost, gradnorm, stepsize);
    end
    
    % Stopping criteria
    if gradnorm  <= options.tolgradnorm
        if options.verbosity > 0
            fprintf('Norm of gradient smaller than %g.\n',options.tolgradnorm);
        end
        break;
    end
    
end

info = info(1:epoch+1);


% Save the stats per epoch.
    function stats = savestats()
        stats.epoch = epoch;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if epoch == 0
            stats.time = 0;
        else
            stats.time = info(epoch).time + elapsed_time;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end

% Save the stats per iteration.
    function innerstats = saveinnerstats()
        innerstats.inneriter = inneriter;
        if inneriter == 0
            innerstats.cost = NaN;
            innerstats.gradnorm = NaN;
            innerstats.time = 0;
        else
            innerstats.cost = cost;
            %innerstats.gradnorm = gradnorm; % HK
            innerstats.gradnorm = partialgradnorm;
            if inneriter == 1
                innerstats.time = elapsed_time;
            else
                innerstats.time = innerinfo(inneriter-1).time + elapsed_time;
            end
        end
        
    end

    function [output, k, nr] = myCG(u, w, b, data, tau, maxiter)
        x = zeros(size(b));
        r = b;% - Axhandle(x);
        p = r;
        k = 0;
        zero = x;
        while(norm(r, 'fro') > tau * norm(b,'fro') && k < maxiter)
            %     while(norm(r, 'fro') > tau * norm(b,'fro') && k < maxiter)
            Ap = zero;
            for i = 1:N
                wi = w(:,i);
                Ap = Ap + data(i).XtX*(p*(wi*wi'));
            end
            Ap = Ap / N;
            Ap = Ap - u*(u'*Ap);
            
            alpha = r(:)' * r(:) / (p(:)' * Ap(:));
            x = x + alpha * p;
            rr0 = r(:)' * r(:);
            r = r - alpha * Ap;
            beta = r(:)' * r(:) / rr0;
            p = r + beta * p;
            k = k + 1;
        end
        nr = norm(r, 'fro');
        %     fprintf("the CG error %e \n",nr);
        output = x;
    end



end

