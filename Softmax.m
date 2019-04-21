function p = Softmax(x)

n = length(x);
sum_exp = sum(exp(x));
p = zeros(n,1);

for i = 1:n
    p(i) = exp(x(i)) / sum_exp;
end