function [y] = ReLu(x)
%Input: x is a general tensor, matrix, and vector.
%Output: y is the output of the Rectified Linear Unit (ReLu) with the same input size.
%Description: ReLu is an activation unit (yi = max(0, xi)). In some case, it is possible to use
%a Leaky ReLu (yi = max(eps*xi, xi) where eps = 0.01).

%y = max(0, x);

% Leaky ReLu
epsilon = 0.01;
y = max(epsilon*x, x);