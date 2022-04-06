function J = update_J(A, lambda)
    [n1, n2] = size(A);
    J = zeros(n1, n2);
    for i = 1:n1
        J(i, :) = update_Ji(A(i,:), lambda);
    end
end

function Ji = update_Ji(a, lambda)
    [~, S] = sort(a, 'descend', 'ComparisonMethod', 'abs');
    d = length(a);
    tau = d;
    mu = sum(abs(a))/d;
    while tau > 1 && abs(a(S(tau))) - (lambda*tau)/(1+lambda*tau)*mu < 0
        mu = tau/(tau - 1)*mu  - 1/(tau - 1)*abs(a(S(tau)));
        tau = tau - 1;
    end
    Ji = sign(a).*max(abs(a)-(lambda*tau)/(1+lambda*tau)*mu, 0);
end

% function Ji = update_Ji(a, lambda)
%     Ji = sign(a).*max(abs(a)-lambda/2, 0);
% end