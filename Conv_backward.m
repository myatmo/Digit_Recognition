function [dLdw, dLdb] = Conv_backward(dLdy, x, w_conv, b_conv, y)
%Input: dLdy is the loss derivative with respec to y.
%Output: dLdw and dLdb are the loss derivatives with respect to convolutional weights
%and bias w and b, respectively.
%Description: This convolutional operation can be simplified using MATLAB built-in function im2col.
%Note that for the single convolutional layer, ?L/?x is not needed.

H = size(dLdy,1);
W = size(dLdy,2);
C = size(dLdy,3);

dLdw = 0;
dLdb = 0;
x_col = im2col(x, [H W]);
% calculate loss derivative w.r.t w
for i = 1:C
    dLdy_col = im2col(dLdy(:,:,i), [H W]);
    dLdw = dLdw + dLdy_col'*x_col;
    
    % calculate loss derivative w.r.t b
    dLdb = dLdb + sum(dLdy_col);
end




