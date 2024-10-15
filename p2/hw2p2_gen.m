clear
close all
clc

x = load('food_dataset_values.csv');

food_features
food_classes

% Extract class labels
clab = x(:,2);
x(:,1:2) = [];

[x, mu, sig] = zscore(x);


[clab table] = fix_cluster_indices(clab);
for c=1:max(clab)
  fprintf('\nclass %2d is [%4d]', c, table(c));
end
fprintf('\n');

%% Remove classes with very few examples
for c=1:max(clab)
  nec(c) = sum(c==clab);
end
ix = [];
for c=1:max(clab)
  if nec(c)<300
    ix = [ix; find(clab==c)];
    fprintf('\nclass %d is out', c);
  end
end
fprintf('\n');

x(ix,:)    = [];
clab(ix,:) = [];
[clab table] = fix_cluster_indices(clab);
for c=1:max(clab)
  fprintf('\nclass %d is [%d]', c, table(c));
end
fprintf('\n');

nc = max(clab);

%% Split into training, validation and test sets

% First split into training (x1) and test (x3) data
[x_ clab_ x3 clab3] = split_dataset(x, clab, .6);

figure
subplot(313); hist(clab3, nc); title('Test samples per class')

if 1

  csvwrite('mu.csv', mu);
  csvwrite('sig.csv', sig);

  % save test_true  x3 clab3
  csvwrite('x3.csv', x3);
  csvwrite('c3_true.csv', clab3);
  
  clab3 = clab3(randperm(length(clab3)));
  % save test_fake  x3 clab3
  csvwrite('c3_fake.csv', clab3);
  
  % Second split into training and validation data
  
  [x1 clab1 x2 clab2] = split_dataset(x_, clab_, .5);
  
  subplot(311); hist(clab1, nc); title('Training samples per class')
  subplot(312); hist(clab2, nc); title('Validation samples per class')
  
  % save train x1 clab1
  % save val  x2 clab2
  
  csvwrite('x1.csv', x1);
  csvwrite('x2.csv', x2);
  csvwrite('c1.csv', clab1);
  csvwrite('c2.csv', clab2);
end