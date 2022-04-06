function W = my_constructW_PKN(X, k, issymmetric)
%% Note: Construct similarity matrix with probabilistic k-nearest neighbors. 
%%       It is a parameter free, distance consistent similarity.
%% Input:
%	X: each column is a data point
%	k: number of neighbors
%	issymmetric: set W = (W+W')/2 if issymmetric=1
%% Outout:
%	W: similarity matrix

    if nargin < 3
        issymmetric = 1;
    end
    if nargin < 2
        k = 5;
    end

    [~, n] = size(X);
    D = L2_distance_1(X, X);
    [~, idx] = sort(D, 2); % sort each row

    W = zeros(n);
    for i = 1:n
        id = idx(i,2:k+2);
        di = D(i, id);
        W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end

    if issymmetric == 1
        W = (W+W')/2;
    end
end


function d = L2_distance_1(a, b)
%% Note: Compute squared Euclidean distance
%%       ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
%% Input:
%   a,b: two matrices. each column is a data
%% Output:
%   d:   distance matrix of a and b

    if (size(a,1) == 1)
      a = [a; zeros(1,size(a,2))]; 
      b = [b; zeros(1,size(b,2))]; 
    end

    aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
    d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

    d = real(d);
    d = max(d,0);
end





