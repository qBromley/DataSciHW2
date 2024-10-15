% Plots the features of matrix x in 2 or 3 dimensions
%
%  plot_scatter(x, clab, dims)
%   x:      database matrix (row vectors)
%   clab:   vector of labels
%   dims:   2 or 3

function plot_scatter(x, clab, dims)

font_size = 10;
colormap('jet')

nc   = num_classes(clab);
cmap = colormap;

for e=1:size(x,1)
  c = clab(e);
  ccolor = cmap(round(clab(e)*size(cmap,1)/nc),:);
  
  if (dims==2)
    text(x(e,1), x(e,2), num2str(clab(e)), ...
      'FontSize', [font_size], ...
      'FontName', 'Helvetica', ...
      'color', ccolor);
    hold on;
    
  else
    text(x(e,1), x(e,2), x(e,3), num2str(clab(e)), ...
      'FontSize', [font_size], ...
      'FontName', 'Helvetica', ...
      'color', ccolor);
  end;
  hold on;
  
end;

box on;
%grid on;

xlabel('axis 1');
ylabel('axis 2');

if (dims==2)
  axis([min(x(:,1)) max(x(:,1)) min(x(:,2)) max(x(:,2))]);
  zoom on;
else
  zlabel('axis 3');
  axis([min(x(:,1)) max(x(:,1)) min(x(:,2)) max(x(:,2)) min(x(:,3)) max(x(:,3))]);
  view([45 45]);
  rotate3d;
end;