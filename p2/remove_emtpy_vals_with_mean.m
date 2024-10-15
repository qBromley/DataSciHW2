function x = remove_emtpy_vals_with_mean(x)

x1 = x;

for c=1:size(x,2)
    ix = find(x(:,c) ~= -1);
    m = mean(x(ix,c));
    ix = find(x(:,c) == -1);
    for i=ix 
        x(i,c)=m;
    end
end