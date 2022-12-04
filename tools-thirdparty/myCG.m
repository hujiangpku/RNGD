function [output, k, nr] = myCG(Axhandle, b, tau, maxiter)
    x = zeros(size(b));
    r = b;% - Axhandle(x);
    p = r;
    k = 0;
%     while(norm(r, 'fro') > tau * min(lambdanFz * norm(x, 'fro'), 1) && k < maxiter)
    while(norm(r, 'fro') > tau * min(norm(b,'fro'), 1e-8) && k < maxiter)
%     while(norm(r, 'fro') > tau * norm(b,'fro') && k < maxiter)
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
%     fprintf("the CG error %e \n",nr);
    output = x;
end
