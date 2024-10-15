% Splits a dataset into 'fract' for testing and '1-fract' for training
% The split is done one class at a time to ensure that (1) priors are
% maintained in the splits and (2) no class ends up with zero samples 
% on either set

function [x1 clab1 x2 clab2] = split_dataset(x,clab, fract)

for c=1:max(clab)
  nec(c) = sum(c==clab);
end
if min(nec)<2
  error('not enough data to split into training and test sets')
end

done = 0;
x1    = [];
clab1 = [];
x2    = [];
clab2 = [];
while ~done
  for c=1:max(clab)
    ixc   = find(clab==c);
    xc    = x(ixc,:);
    clabc = clab(ixc,:);

    r = rand(size(xc,1),1);

    x1    = [x1;    xc(find(r>fract),:)];
    clab1 = [clab1; clabc(find(r>fract),:)];

    x2    = [x2;    xc(find(r<=fract),:)];
    clab2 = [clab2; clabc(find(r<=fract),:)];

  end
  for c=1:max(clab)
    nec1(c) = sum(c==clab1);
    nec2(c) = sum(c==clab2);
  end
  if min(nec1)>0 & min(nec2)>0
    done = 1;
  else
      %     fprintf('\nsplit_dataset*');
  end
end
  
% Shuffle rows
ne = size(x1,1);
rows = randperm(ne);
x1 = x1(rows,:);
clab1 = clab1(rows,:);

ne = size(x2,1);
rows = randperm(ne);
x2 = x2(rows,:);
clab2 = clab2(rows,:);
