% Perform sequential forward selection on a classification dataset

function [best_subset, best_score] = sfs(x, clab, obf_name)

nf = size(x,2);

full_set = 1:nf;
selected = zeros(1,nf);

best_subset = zeros(1,nf);
best_score = -Inf;

%while sum(selected==0)>0
hist = [];
for it=1:10 %nf
  obf_val = zeros(1,nf);
  for ix = find(selected==0)
    x1 = x(:,[find(selected==1) ix]);
    obf_val(ix) = eval([obf_name '(x1,clab);']);
  end
  
  [maxval maxix] = max(obf_val);
  selected(maxix) = 1;
  fprintf('sfs [%2d]: feature #%2d was selected [%2f]\n', it, maxix, maxval);
  
  if maxval > best_score
    best_score  = maxval;
    best_subset = selected;
  end
  hist = [hist maxval];
end

plot(hist)