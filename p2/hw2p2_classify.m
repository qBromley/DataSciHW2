clear
close all
clc

x1    = csvread('x1.csv');
clab1 = csvread('c1.csv');
x1    = remove_emtpy_vals_with_mean(x1);

% x2    = csvread('x2.csv');
% clab2 = csvread('c2.csv');
% x2 = remove_emtpy_vals_with_mean(x2);

x3    = csvread('x3.csv');
clab3 = csvread('c3_true.csv');
x3    = remove_emtpy_vals_with_mean(x3);

[y v d] = tamu_lda(x1, clab1);
y1 = x1*v;
y3 = x3*v;
[uclass] = tamu_knn_class(y1, clab1, y3, 1);
score    = tamu_class_rate(uclass,clab3);
fprintf('KNN C_RATE ON FULL FEATURE SET: %2f\n\n\n', score);

[best_subset, best_score] = sbs(x1, clab1, 'fss_wrapper');

x1 = x1(:, find(best_subset));
x3 = x3(:, find(best_subset));

[y v d] = tamu_lda(x1, clab1);
y1 = x1*v;
y3 = x3*v;
[uclab] = tamu_knn_class(y1, clab1, y3, 1);
score    = tamu_class_rate(uclab,clab3);
fprintf('KNN C_RATE ON REDUCED FEATURE SET: %2f\n', score);

uclab = uclab';

csvwrite('rgo_solution.csv', uclab);

