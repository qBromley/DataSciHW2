% Perform sequential backward selection on a classification dataset

function [best_subset, best_score] = sbs(x, clab, obf_name)

nf = size(x,2);

full_set = 1:nf;
selected = ones(1,nf);

best_subset = ones(1,nf);
best_score = -Inf;

for it=1:10
%while sum(selected==1)>0
  obf_val = zeros(1,nf);
  for ix = find(selected==1)
    selected1     = selected;
    selected1(ix) = 0;
    x1 = x(:,[find(selected1==1) ix]);
    obf_val(ix) = eval([obf_name '(x1,clab);']);
  end
  
  [maxval maxix] = max(obf_val);
  selected(maxix) = 0;
  fprintf('sfs: feature #%2d was removed [%2f]\n', maxix, maxval);
  
  if maxval >best_score
    best_score  = maxval;
    best_subset = selected;
  end
end

