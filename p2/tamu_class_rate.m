function rate = class_rate(pred_class, true_class)

if isempty(pred_class) | isempty(true_class)
  rate = 0;
  return;
end;

pred_class = reshape(pred_class, [length(pred_class) 1]);
true_class = reshape(true_class, [length(true_class) 1]);

rate = sum(pred_class==true_class)/length(true_class);

%fprintf('class_rate(): %6.2f\n', 100*rate);
