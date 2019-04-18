function [dLdx] = ReLu_backward(dLdy, x, y)
%Input: dLdy ? R1×z is the loss derivative with respect to the output y ? Rz where z is
%the size of input (it can be tensor, matrix, and vector).
%Output: dLdx ? R1×z is the loss derivative with respect to the input x.

epsilon = 0.01;
n = length(x);

dLdx = zeros(1,n);
for i=1:n
    % calculate derivative of L w.r.t x
    if y(i) >= 0
        dLdx(i) = dLdy(i);
    else
        dLdx(i) = epsilon;
    end
end




