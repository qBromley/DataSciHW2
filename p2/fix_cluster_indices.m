% Reassign cluster indices so that they are consecutive numbers starting at 1
%
% [cix1] = fix_cluster_indices(cix)
%      cix1:    fixed cluster indices
%      cix:     original cluster indices

function [cix1, table] = fix_cluster_indices(cix)

flags = zeros(max(cix), 1);
cnt   = 0;

for e=1:size(cix,1)
  if flags(cix(e))==0 % first time for cix(e)
    cnt           = cnt+1;
    flags(cix(e)) = cnt;
    table(cnt,1)  = cix(e);
  end;
end;

for e=1:size(cix,1)
  cix1(e,1) = flags(cix(e));
end;
