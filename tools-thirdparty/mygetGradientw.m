function [grad, w] = mygetGradientw(problem, x, storedb, key)
% Computes the gradient of the cost function at x.
%
% function grad = getGradient(problem, x)
% function grad = getGradient(problem, x, storedb)
% function grad = getGradient(problem, x, storedb, key)
%
% Returns the gradient at x of the cost function described in the problem
% structure.
%
% storedb is a StoreDB object, key is the StoreDB key to point x.
%
% See also: getDirectionalDerivative canGetGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%  June 28, 2016 (NB):
%       Works with getPartialGradient.
%
%   Nov. 1, 2016 (NB):
%       Added support for gradient from directional derivatives.
%       Last resort is call to getApproxGradient instead of an exception.

    % Allow omission of the key, and even of storedb.
    if ~exist('key', 'var')
        if ~exist('storedb', 'var')
            storedb = StoreDB();
        end
        key = storedb.getNewKey();
    end

    
    if isfield(problem, 'gradw')
    %% Compute the gradient using grad.
	
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.grad)
            case 1
                [grad, w] = problem.grad(x);
            case 2
                % Obtain, pass along, and save the store for x.
                store = storedb.getWithShared(key);
                [grad, w, store] = problem.grad(x, store);
                storedb.setWithShared(store, key);
            case 3
                % Pass along the whole storedb (by reference), with key.
                [grad, w] = problem.grad(x, storedb, key);
            otherwise
                up = MException('manopt:getGradient:badgrad', ...
                    'grad should accept 1, 2 or 3 inputs.');
                throw(up);
        end
    
    elseif isfield(problem, 'costgradw')
    %% Compute the gradient using costgrad.
		
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.costgradw)
            case 1
                [unused, grad, w] = problem.costgradw(x); %#ok
            case 2
                % Obtain, pass along, and save the store for x.
                store = storedb.getWithShared(key);
                [unused, grad, w, store] = problem.costgradw(x, store); %#ok
                storedb.setWithShared(store, key);
            case 3
                % Pass along the whole storedb (by reference), with key.
                [unused, grad, w] = problem.costgradw(x, storedb, key); %#ok
            otherwise
                up = MException('manopt:getGradient:badcostgrad', ...
                    'costgrad should accept 1, 2 or 3 inputs.');
                throw(up);
        end
    
    elseif canGetEuclideanGradientw(problem)
    %% Compute the gradient using the Euclidean gradient.
        
        [egrad, w] = mygetEuclideanGradientw(problem, x, storedb, key);
        grad = problem.M.egrad2rgrad(x, egrad);
    
    elseif canGetPartialGradientw(problem)
    %% Compute the gradient using a full partial gradient.
        
        [grad, w] = mygetPartialGradientw(problem, x, 1:d, storedb, key);
        
    else
        fprintf('please at least input problems Euclidean gradient !');
        exit;
    end
    
end
