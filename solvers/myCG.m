function [output, k, nr] = myCG(Axhandle, b, tau, maxiter)
% Using trunctated conjugate gradient method to inexactly solve a linear system Axhandle(output) = b, 
% where Axhandle is a defined linear operator. tau is the parapeter to control the inexactness. 
% maxiter represents the maximal number of CG iterations.

    x = zeros(size(b));
    r = b;% - Axhandle(x);
    p = r;
    k = 0;
    while(norm(r, 'fro') > tau * min(norm(b,'fro'), 1e-8) && k < maxiter)
        Ap = Axhandle(p);
        alpha = r(:)' * r(:) / (p(:)' * Ap(:));
        x = x + alpha * p;
        rr0 = r(:)' * r(:);
        r = r - alpha * Ap;
        beta = r(:)' * r(:) / rr0;
        p = r + beta * p;
        k = k + 1;
    end
    nr = norm(r, 'fro');
    output = x;
end
