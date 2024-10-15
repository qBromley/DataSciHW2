function [uclass, ucp] = cs790_knn_class(x, clab, xu, k)

if isempty(x) | isempty(xu)
  uclass = [];
  ucp = [];
  return;
end;

nc = max(clab);

if k==0
  k = min(1, round(min(num_ex_class(xd))/2));
end;

% Compute distances

d = dist(x, xu');

% Sort distances and find K closest samples

[sd si] = sort(d);
for c=1:nc
  cneigh = (clab(si(1:k,:))==c);
  if k>1
    ucp(c,:) = sum( cneigh );
  else 
    % For k==1, 'cneigh' becomes a vector, so MATLAB-adding will collapse it.
    % Matlab "seems" to transpose 'cneigh' to a column so we reshape to a row.
    ucp(c,:) = reshape(cneigh, 1,length(cneigh));
  end
end;
ucp = ucp/k;

[maxucp, uclass] = max(ucp);
