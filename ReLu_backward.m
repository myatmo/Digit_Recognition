function [dLdx] = ReLu_backward(dLdy, x, y)
%Input: dLdy ? R1×z is the loss derivative with respect to the output y ? Rz where z is
%the size of input (it can be tensor, matrix, and vector).
%Output: dLdx ? R1×z is the loss derivative with respect to the input x.

epsilon = 0.01;
H = size(dLdy,1);
W = size(dLdy,2);
C = size(dLdy,3);

%dLdx = zeros(1,n);
for i=1:C
    %x_col = reshape(x(:,:,i),[],1)
    y_col = reshape(y(:,:,i),[],1);
    f = zeros(1,length(y_col));
    for j=1:length(y_col)
        % calculate derivative of L w.r.t x
        if y_col(j) >= 0
            f(j) = dLdy(j);
        else
            f(j) = epsilon;
        end
    end
    if H > 1
        f = reshape(f,[H W]);
        dLdx(:,:,i) = f;
    else
        dLdx = f;
    end
end




