function [grad, w] = mygetPartialGradientw(problem, x, I, storedb, key)
% Computes the gradient of a subset of terms in the cost function at x.
%
% function grad = getPartialGradient(problem, x, I)
% function grad = getPartialGradient(problem, x, I, storedb)
% function grad = getPartialGradient(problem, x, I, storedb, key)
%
% Assume the cost function described in the problem structure is a sum of
% many terms, as
%
%    f(x) = sum_i f_i(x) for i = 1:d,

% where d is specified as d = problem.ncostterms.
% 
% For a subset I of 1:d, getPartialGradient obtains the gradient of the
% partial cost function
% 
%    f_I(x) = sum_i f_i(x) for i = I.
%
% storedb is a StoreDB object, key is the StoreDB key to point x.
%
% See also: getGradient canGetPartialGradient getPartialEuclideanGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 28, 2016
% Contributors: 
% Change log: 


    % Allow omission of the key, and even of storedb.
    if ~exist('key', 'var')
        if ~exist('storedb', 'var')
            storedb = StoreDB();
        end
        key = storedb.getNewKey();
    end
    
    
    % Make sure I is a row vector, so that it is natural to loop over it
    % with " for i = I ".
    I = (I(:)).';

    
    if isfield(problem, 'partialgradw')
    %% Compute the partial gradient using partialgrad.
	
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.partialgradw)
            case 2
                [grad, w] = problem.partialgradw(x, I);
            case 3
                % Obtain, pass along, and save the store for x.
                store = storedb.getWithShared(key);
                [grad, w, store] = problem.partialgradw(x, I, store);
                storedb.setWithShared(store, key);
            case 4
                % Pass along the whole storedb (by reference), with key.
                [grad, w] = problem.partialgradw(x, I, storedb, key);
            otherwise
                up = MException('manopt:getPartialGradient:badpartialgrad', ...
                    'partialgrad should accept 2, 3 or 4 inputs.');
                throw(up);
        end
    
    elseif canGetPartialEuclideanGradient(problem)
    %% Compute the partial gradient using the Euclidean partial gradient.
        
        [egrad, w] = mygetPartialEuclideanGradientw(problem, x, I, storedb, key);
        grad = problem.M.egrad2rgrad(x, egrad);

    else
    %% Abandon computing the partial gradient.
    
        up = MException('manopt:getPartialGradient:fail', ...
            ['The problem description is not explicit enough to ' ...
             'compute the partial gradient of the cost.']);
        throw(up);
        
    end
    
end
