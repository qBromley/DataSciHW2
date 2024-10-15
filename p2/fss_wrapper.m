function [score] = fss_wrapper(x,clab)

ne = size(x,1);

nrep = 10;

% score = 0;
score = [];
for rep=1:nrep
  [x1 clab1 x2 clab2] = split_dataset(x,clab, .2);
  k = 5;
  if 1
    [y v d] = tamu_lda(x1, clab1);
    y1 = x1*v;
    y2 = x2*v;
    [uclass] = tamu_knn_class(y1, clab1, y2, k);
  else
    [uclass] = tamu_knn_class(x1, clab1, x2, k);
    %[uclass] = tamu_quad_class(x1, clab1, x2);
  end
%   score    = score + class_rate(uclass,clab2);
score = [score; class_rate(uclass,clab2)];
end
%score = score/nrep;
score = mean(score);
