clear
close all
clc

x1    = csvread('x1.csv');
clab1 = csvread('c1.csv');
x1    = remove_emtpy_vals_with_mean(x1);

x2    = csvread('x2.csv');
clab2 = csvread('c2.csv');
x2 = remove_emtpy_vals_with_mean(x2);

% x3    = csvread('x3.csv');
% clab3 = csvread('c3_true.csv');
% x3    = remove_emtpy_vals_with_mean(x3);

%% PCA

[y v d] = tamu_pca(x1);
y1 = x1*v;
y2 = x2*v;

plot_scatter(y1, clab1, 2)

figure
plot(diag(d), 'o-')

%% LDA

figure
[y v d] = tamu_lda(x1, clab1);
y1 = x1*v;
y2 = x2*v;

plot_scatter(y1, clab1, 2)

figure
plot(d, 'o-')
