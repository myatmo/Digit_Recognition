function y = FC(x, w, b)
%Input: x ? Rm is the input to the fully connected layer, and w ? Rn×m and b ? Rn are
%the weights and bias.
%Output: y ? Rn is the output of the linear transform (fully connected layer).
%Description: FC is a linear transform of x, i.e., y = wx + b.

y = w*x + b;