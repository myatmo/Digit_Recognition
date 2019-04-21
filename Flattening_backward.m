function [dLdx] = Flattening_backward(dLdy, x, y)
%Input: dLdy is the loss derivative with respect to the output y.
%Output: dLdx is the loss derivative with respect to the input x.

H = size(x,1);
W = size(x,2);
C = size(x,3);

y1 = dLdy' .* y; % don't think this is needed!
for i = 1:C
    s = ((i-1)*length(y)/C) + 1; % the starting index of the vectorized tensor
    e = (length(y)/C) * i; % the ending index
    dLdx(:,:,i) = reshape(y1(s:e),H,W); % reshape it into HxW in each dimension C
end


