% Computes the Principal Components of the multivariate data 'x'
% Results are returned sorted by the magnitude of the eigenvalues: 
%    d(1,1) > d(2,2) > d(3,3) ...    
%
%   [y, v, d] = pca(x, varqargin)
%      y:       projected data, y = x*v
%      v:       eigenvectors matrix (column vectors)
%      d:       eigenvalue matrix:   d = v'*cov(x)*v
%      x:       database matrix (row vector)

function [y, v, d] = pca(x)

%fprintf('hw1p2 pca()...\t');

nd = size(x,2);

cvar = cov(x);
[v1, d1] = eig(cvar);

% sort 'v' and 'd' by decreasing eigenvalue

v = sort_feats(v1, diag(d1));
d = zeros(size(d1));
eigvalue = -sort(diag(-d1)); % 'sort' returns increasing values
for k=1:nd
  d(k,k) = eigvalue(k);
end;

y=x*v;

%fprintf('...done\n');

% Rearranges the features (columns) of matrix 'x' in ***DECREASING*** order 
% according to score 'sc'
%
% [y] = sort_feats(x, sc)
%   x:        database matrix (column vectors)
%   sc:       score of each column

function [y] = sort_feats(x, sc)

y = zeros(size(x));
[sort_sc min_ix] = sort(sc);

for f=1:size(x,2)
  y(:,size(x,2)+1-f) = x(:,min_ix(f));
end;
