% Returns the number of different classes in the database vector xd
%
%  nc = num_classes(clab)
%   nc:       number of classes
%   xd:       vector of labels

function nc = num_classes(clab)

tmp = zeros(size(clab));

for k=1:length(clab)
  tmp(clab(k)) = 1;
end;

nc = sum(tmp);