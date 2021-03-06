function [w, b] = TrainSLP(mini_batch_x, mini_batch_y)
%Output: w ? R10�196 and b ? R10�1 are the trained weights and bias of a single-layer perceptron.
%Description: You will use the following functions to train a single-layer perceptron using
%a stochastic gradient descent method: FC, FC_backward, Loss_cross_entropy_softmax.

m = size(mini_batch_x{1},1); % size of each image which is 196
n = size(mini_batch_y{1},1); % size of each label (one-hot encoding) which is 10

% Set the learning rate and the decay rate ? (0, 1]
gamma = 0.01;
lambda = 0.6;

% Initialize the weights with a Gaussian noise w ? N (0, 1) and bias
w = rand([n,m])*sqrt(2/(n+m));
b = rand([n,1])*sqrt(2/(n));

k = 1; % initialize k=1
nIters = 10000;
for iter = 1:nIters
    iter
    if mod(iter,1000) == 0 %6: at every 1000th iteration, ? ? ??
        gamma = lambda*gamma;
    end
    
    % Set dL/dw and dL/db to 0
    dLdw = 0;
    dLdb = 0;
    
    % Loop for each image xi in kth mini-batch
    batch_size = size(mini_batch_x{k},2); % num of images in kth mini_batch
    for i = 1:batch_size
        x = mini_batch_x{k}(:,i); % each image in kth mini_batch
        y_tilde = FC(x, w, b); % label prediction of xi
        
        y = mini_batch_y{k}(:,i); % ground truth label
        [l, dldy] = Loss_cross_entropy_softmax(y_tilde, y); % compute cross entropy loss l
        
        [dldx, dldw, dldb] = FC_backward(dldy, x, w, b, y_tilde); % back-propagation of xi, dl/dw using back-propagation
        
        % Update dLdw and dLdb
        dLdw = dLdw + dldw;
        dLdb = dLdb + dldb;
    end
    % Set k = 1 if k is greater than the number of mini-batches else k++
    if k >= length(mini_batch_x)
        k = 1;
    else
        k = k + 1;
    end
    
    % Update the weights and bias
    w = w - ((gamma/batch_size) * reshape(dLdw,[size(dLdw,2),size(dLdw,3)]));
    b = b - ((gamma/batch_size) * dLdb');
end


