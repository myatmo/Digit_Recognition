function [y] = Pool2x2(x)
%Input: x ? RH×W×C is a general tensor and matrix.
%Output: y ? RH/2 × W/2 × C is the output of the 2 × 2 max-pooling operation with stride 2.

H = size(x,1);
W = size(x,2);
C = size(x,3);

y = zeros(H/2,W/2,C);

for i = 1:C
    x_col = im2col(x(:,:,i),[2,2],'distinct');
    f = zeros(size(x_col,2),1);
    for j = 1:size(x_col,2)
        f(j) = max(x_col(:,j)); % get max and add it to f
    end
    
    f = reshape(f,H/2,W/2); % reshape f
    y(:,:,i) = f;
end



