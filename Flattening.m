function [y] = Flattening(x)
%Input: x ? RH×W×C is a tensor.
%Output: y ? R HWC is the vectorized tensor (column major).

H = size(x,1);
W = size(x,2);
C = size(x,3);

y = [];
for i = 1:C
    y = [y; reshape(x(:,:,i),[],1)];
end



