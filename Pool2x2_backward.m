function [dLdx] = Pool2x2_backward(dLdy, x, y)
%Input: dLdy is the loss derivative with respect to the output y.
%Output: dLdx is the loss derivative with respect to the input x

H = size(x,1);
W = size(x,2);
C = size(x,3);

for i = 1:C
    x_col = im2col(x(:,:,i),[2,2],'distinct');
    dLdy_col = reshape(dLdy(:,:,i),[],1);
    y_col = reshape(y(:,:,i),[],1) .* dLdy_col; % multiply the y value with the gradient
    f = zeros(size(x_col,1),size(x_col,2)); % construct a zero arry of the same size as x_col
    for j = 1:size(x_col,2)
        [~,idx] = max(x_col(:,j)); % get the index of max value
        f(idx,j) = y_col(j);
    end
    f = col2im(f,[2 2],[size(x,1) size(x,2)],'distinct');
    dLdx(:,:,i) = f;
end



