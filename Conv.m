function [y] = Conv(x, w_conv, b_conv)
%Input: x ? RH×W×C1 is an input to the convolutional operation, w_conv ? RH×W×C1×C2 and
%b_conv ? RC2 are weights and bias of the convolutional operation.
%Output: y ? RH×W×C2 is the output of the convolutional operation. Note that to get the same size
%with the input, you may pad zero at the boundary of the input image.
%Description: This convolutional operation can be simplified using MATLAB built-in function im2col.

H = size(x,1);
W = size(x,2);
C2 = length(b_conv);
H_wconv = size(w_conv,1);
W_wconv = size(w_conv,2);

pad_x = padarray(x,[1,1],0); % pad 0 around the image
% rearrange x and w_conv values into a column-wise arrangement
x_col = im2col(pad_x, [H_wconv W_wconv]);

y = zeros(H,W,C2);
for i = 1:C2
    w_conv_col = im2col(w_conv(:,:,i),[H_wconv W_wconv]);
    w_conv_col = w_conv_col';
    f = w_conv_col * x_col;
    f = reshape(f',H,W);
    y(:,:,i) = f;
end





