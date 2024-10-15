function [score] = fss_filter(x,clab)

[cmean, ccov] = class_stats(x, clab);
[nc nd]       = size(cmean);

for c1=1:nc
  x1 = x(find(clab==c1),:);
  for c2=1:nc
    if size(x,2)==1
      dclass(c1,c2) = mean( (x1-cmean(c2,:))/ccov(c2) );
    else
      dclass(c1,c2) = mean( mahdist(x1, cmean(c2,:)', reshape(ccov(c2,:), nd, nd)) );
    end
  end
end

score = mean(mean(dclass));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Computes the mean and covariance matrix for each class in the database
%   
%   [cmean, ccov] = class_stats(x, clab)
% 
%     cmean:    class means (row vectors)
%     ccov:     class covariances (stacked by rows)
%     x:        database matrix
%     clab:       database dimensions

function [cmean, ccov] = class_stats(x, clab)

nc = max(clab);
nd = size(x,2);

for c=1:nc
  xc = x(find(clab==c),:);
  cmean(c,:) = mean(xc);
  ccov(c,:)  = reshape(cov(xc), 1, nd*nd);
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
