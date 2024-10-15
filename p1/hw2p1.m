clear
close all
clc

if 1  % generate data
  
  %% Generate predictors
  
  n   = 100;    % number of examples
  d   = 5;       % number of predictors
  dd  = 20;      % number of noisy channels
  
  x = randn(n,d);        % normally distributed predictors (IVs)
  x = [ones(n,1) x];     % add intercept
  w = randn(d+1,1);      % forward model (from x to y)
  w = w/sqrt(sum(w.^2)); % normalize regression coeffs (unit length)
  y = x*w;               % generate dependent variable (DV)
  
  std = .2;                      % standard dev of additive noise
  y   = y + std*randn(size(y));  % add noise to DV
  
  x = [x randn(n,dd)]; % Add noisy channels
  
  ix = randperm(size(x,2));  % shuffle predictors and noise channels
  x  = x(:,ix);              % predictors are no longer on columns 1:d+1
  
  %% Split data
  
  x1 = x(1:(n/2),:);  % training data
  y1 = y(1:(n/2),:);
  
  x2 = x((n/2+1):end,:); % test data
  y2 = y((n/2+1):end,:);
  
  csvwrite('x1.csv', x1); % save data on CSV format
  csvwrite('x2.csv', x2);
  csvwrite('y1.csv', y1);
  csvwrite('y2.csv', y2);
end

x1 = csvread('x1.csv');
x2 = csvread('x2.csv');
y1 = csvread('y1.csv');
y2 = csvread('y2.csv');

x = [x1;x2];
y = [y1; y2];

for rep=1:200
  
  ix = randperm(size(x,1));
  
  x = x(ix,:);
  y = y(ix,:);
  
  x1 = x(1:(n/2),:);  % training data
  y1 = y(1:(n/2),:);
  
  x2 = x((n/2+1):end,:); % test data
  y2 = y((n/2+1):end,:);
  
  
  %%  perform least squares regression
  
  w_pred  = inv(x1'*x1)*x1'*y1;  % pseudo-inverse solution
  
  y1_pred = x1*w_pred;            % predict DV on training data
  
  y2_pred = x2*w_pred;            % predict DV on test     data
  
  rss = sum((y2 - y2_pred).^2);
  tss = sum((y2-mean(y2)).^2);
  r2(rep) = 1 - rss/tss;
  
  
  plot(y1, y1_pred, 'b.')          % plot ground truth vs predicted DV
  hold on
  plot(y2, y2_pred, 'r.')          % plot ground truth vs predicted DV
end
mean(r2)
figure
hist(r2,25)