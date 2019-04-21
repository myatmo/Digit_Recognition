function [dLdx, dLdw, dLdb] = FC_backward(dLdy, x, w, b, y)
%Input: dLdy ? R1×n is the loss derivative with respect to the output y.
%Output: dLdx ? R1×m is the loss derivative with respect the input x, dLdw ? R1×(n×m) is the loss derivative
%with respect to the weights, and dLdb ? R1×n is the loss derivative with respec to the bias.
%Description: The partial derivatives w.r.t. input, weights, and bias will be computed. dLdx will be back-propagated,
%and dLdw and dLdb will be used to update the weights and bias.

m = length(x); % length of the image which is 196
n = length(y); % length of the label (one-hot encoding) which is 10

% calculate partial derivative of loss function w.r.t input x
dydx = w; % partial derivative of y w.r.t x
dLdx = dLdy * dydx;

% calculate partial derivative of loss function w.r.t weights w
Xr = repmat(x', 1, n); % repeat the matrix
X = mat2cell(Xr, size(x',1), repmat(size(x',2),1,n)); % create cell array of repeated matrix X
dydw = blkdiag(X{:}); % partial derivative of y w.r.t w; reshape the array x into matrix of nx(nxm)
dLdw = zeros(1,n,m);
dLdw_hat = dLdy * dydw;
dLdw(1,:,:) = reshape(dLdw_hat,m,n)';
%dLdw(1,:,:) = dLdy' * dydw';

% calculate partial derivative of loss function w.r.t bias b
dydb = 1; % partial derivative of y w.r.t b
dLdb = dLdy' * dydb;
dLdb = dLdb';


