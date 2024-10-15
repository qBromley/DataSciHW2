% Quadratic classifier. Returns the winning class and all class 
% probabilities for an unknown set of samples
%
% function [uclass, ucp] = quad_uva(x, clab, xu)
%    uclass:    winning class for each unknown (a column vector)
%    ucp:       Posterior class probabilities for unknowns (each unknown in a row)
%    x:         database matrix (row vectors)
%    clab:      class labels for training data (column vector)
%    xu:        unknown samples (row vectors)

function [uclass, ucp] = quad_class(x, clab, xu)

if isempty(x) | isempty(xu)
  uclass = [];
  ucp = [];
  return;
end;

[ne nd] = size(x);
nue     = size(xu,1);
nc      = max(clab);

% Compute number of examples per class in the training set
for c=1:nc
  nec(c) = sum(clab==c);
end


% Compute class statistics
[cmean, ccov, tcov] = cstats(x, clab);
if min(nec)<=nd
  % QUAD becomes a linear discriminant function
  fprintf('\tquad(): ndim [%d] < nec [%d]. Using total cov matrix...\n', nd, min(nec));
  ccov = repmat(tcov, nc, 1);
end

% Compute discriminant functions for each class
for c=1:nc
  cmean1 = cmean(c,:);
  ccov1  = ccov((1+(c-1)*nd):(c*nd),:);
  mahd1  = mahdist(xu, cmean1', ccov1);
  
  g(:,c)    = -.5*mahd1.^2 -.5*log(det(ccov1)) + log(nec(c)/sum(nec));
  ucp(:,c) = 1/((2*pi)^(nd/2)*det(ccov1)^.5) * exp(-.5*mahd1.^2)* nec(c)/sum(nec) ;
end;

% Normalize posterior probability
ucp = ucp ./ repmat(sum(ucp,2)+(sum(ucp,2)==0) , 1, nc);

[dummy uclass] = max(g');
uclass = uclass';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computes the mean and covariance matrix for each class in the database
%  
%  [cmean, ccov, tcov] = cstats(x, clab)
%
%    cmean:    class means (row vectors)
%    ccov:     class covariances (stacked by rows)
%    tcov:     total covariance 
%    x:         database matrix (row vectors)
%    clab:      class labels for training data
function [cmean, ccov, tcov] = cstats(x, clab)

nc      = max(clab);
[ne nd] = size(x);

% Compute class conditional mean and covariance
for c=1:nc
  xc         = x(find(clab==c),:);
  cmean(c,:) = mean(xc);
  
  lox             = 1+(c-1)*nd;
  hix             = lox+nd-1;
  ccov(lox:hix,:) = rcov(xc);
end;

% Compute total covariance
tcov = rcov(x);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computes the covariance matrix of a dataset, applying a regularization
% term if the matrix is near-singular
%     covm = rcov(x)
%        x: matrix of row vectors
function [covm] = rcov(x)

ne = size(x,1);

reg_thresh = eps;
g          = 1e-3;

x1   = x-repmat(mean(x), size(x,1), 1); 
covm = x1'*x1/(ne-1);
if rcond(covm) < reg_thresh % Regularize
  fprintf('\trcov(): regularizing ...\n');
  covm = covm*(1-g) + eye(size(covm))*mean(diag(covm))*g;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computes the Mahalanobis distance between vectors using a
% specified covariance matrix.
%
%     mahd = mahdist(xm, tm, covm)
%        xm:     (ne*nd) matrix of row vectors
%        tm:     (nd*nt) matrix of column vectors
%        covm:   (nd*nd) covariance matrix
%        mahd:   (ne*nt) distance matrix

function [mahd] = mahdist(xm, tm, covm)

[nx, ndx] = size(xm);
[ndt, nt] = size(tm);

if (ndx ~= ndt)
  error('Matrix sizes do not match.');
end;

mahd = zeros(nx, nt);
icovm = pinv(covm);
for t=1:nt
  vdiff = xm - repmat(tm(:,t)', [nx 1]);
  tdist = vdiff * icovm .* vdiff;
  mahd(:,t) = sum(tdist')' .^ .5;
end


