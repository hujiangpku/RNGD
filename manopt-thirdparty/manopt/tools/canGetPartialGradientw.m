function candoit = canGetPartialGradientw(problem)
% Checks whether the partial gradient can be computed for a given problem.
% 
% function candoit = canGetPartialGradient(problem)
%
% Returns true if the partial gradient of the cost function can be computed
% given the problem description, false otherwise.
%
% See also: getPartialGradient canGetPartialEuclideanGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 28, 2016.
% Contributors: 
% Change log: 

    candoit = (isfield(problem, 'partialgradw') && isfield(problem, 'ncostterms')) || ...
              canGetPartialEuclideanGradientw(problem);
    
    if isfield(problem, 'partialgradw') && ~isfield(problem, 'ncostterms')
        warning('manopt:partialgrad', ...
               ['If problem.partialgradw is specified, indicate the number n\n' ...
                'of terms in the cost function with problem.ncostterms = n.']);
    end
    
end
