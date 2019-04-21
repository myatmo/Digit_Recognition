function [L, dLdy] = Loss_cross_entropy_softmax(x, y)
%Input: x ? Rm is the input to the soft-max, and y ? 0,1^m is the ground truth label.
%Output: L ? R is the loss, and dLdy is the loss derivative with respect to y.
%Description: Loss_cross_entropy_softmax measure cross-entropy between two distributions
%L = sum(i to m) yi*log(yi_hat) where yi_hat is the soft-max output that approximates the max
%operation by clamping x to [0, 1] range: yi_hat = e^(xi)/sum(i) e^(xi), where xi is the ith element of x.

n = length(x);
L = 0;
sum_exp = sum(exp(x));

dy_hatdx = zeros(n);
y_hat = zeros(n,1);

for i = 1:n
    y_hat(i) = exp(x(i)) / sum_exp;
    L = L + (y(i) * log(y_hat(i)));
end

dLdy = (y_hat - y)';



