function x = update_sigma(sigma, lambda)

if lambda <= 4
    x = roots([1, -sigma, 2*lambda+1, -sigma]);
    x = x(x==real(x));
else
    delta = sqrt(lambda*(lambda-4)^3);
    sigma1 = (lambda^2+10*lambda-2 - delta)/2;
    sigma2 = (lambda^2+10*lambda-2 + delta)/2;
    if sigma^2 >= sigma1 && sigma^2 <= sigma2
        x = roots([1, -sigma, 2*lambda+1, -sigma]);
        y = func_logdet(x, sigma, lambda);
        [~,ind] = min(y);
        x = x(ind);
    else
        x = roots([1, -sigma, 2*lambda+1, -sigma]);
        x = x(x==real(x));
    end
end

% x = roots([1, -sigma, 2*lambda+1, -sigma]);
% y = func_logdet(x, sigma, lambda);
% [~,ind] = min(y);
% x = x(ind);


end


function y = func_logdet(omega, sigma, lambda)
    y = (sigma-omega).^2/2 + lambda*log(1+omega.^2);
end