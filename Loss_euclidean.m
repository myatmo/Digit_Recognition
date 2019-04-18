function [L, dLdy] = Loss_euclidean(y_tilde, y)
%Input: y_tilde ? Rm is the prediction, and y ? 0,1^m is the ground truth label.
%Output: L ? R is the loss, and dLdy is the loss derivative with respect to the prediction.
%Description: Loss_euclidean measures Euclidean distance L = ||y - y^||^2.

L = (norm(y_tilde - y))^2;
dLdy = 2*(y_tilde - y);
dLdy = dLdy';