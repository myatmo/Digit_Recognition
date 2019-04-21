function [w1, b1, w2, b2] = TrainMLP(mini_batch_x, mini_batch_y)
%Output: w1 ? R30×196, b1 ? R30×1, w2 ? R10×30, b2 ? R10×1 are the trained weights and biases of
%a multi-layer perceptron.
%Description: You will use the following functions to train a multi-layer perceptron using
%a stochastic gradient descent method: FC, FC_backward, ReLu, ReLu_backward, Loss_cross_entropy_softmax.
%As a result of training, the network should produce more than 90% of accuracy on the testing data.

m = size(mini_batch_x{1},1); % size of each image which is 196
n = size(mini_batch_y{1},1); % size of each label (one-hot encoding) which is 10

% Set the learning rate and the decay rate ? (0, 1]
gamma = 0.01;
lambda = 0.6;

% Initialize the weights with a Gaussian noise w ? N (0, 1) and bias
w1 = rand([n,m])*sqrt(2/(n+m));
w2 = rand([n,n])*sqrt(2/(n+n));
b1 = rand([n,1])*sqrt(2/(n));
b2 = rand([n,1])*sqrt(2/(n));

k = 1; % initialize k=1
nIters = 10000;
for iter = 1:nIters
    iter
    if mod(iter,1000) == 0 %6: at every 1000th iteration, ? ? ??
        gamma = lambda*gamma;
    end
    
    % Set dL/dw and dL/db to 0
    dLdw1 = 0; dLdw2 = 0;
    dLdb1 = 0; dLdb2 = 0;
    
    % Loop for each image xi in kth mini-batch
    batch_size = size(mini_batch_x{k},2); % num of images in kth mini_batch
    for i = 1:batch_size
        x = mini_batch_x{k}(:,i); % each image in kth mini_batch
        y = mini_batch_y{k}(:,i); % ground truth label
        k
        i
        a1 = FC(x, w1, b1); % label prediction of xi for the first layer
        f1 = ReLu(a1); % ReLu
        
        a2 = FC(f1, w2, b2); % label prediction of xi for the second layer
        f2 = ReLu(a2); % ReLu
        
        [l, dldf2] = Loss_cross_entropy_softmax(f2, y); % compute cross entropy loss l for f2
        dlda2 = ReLu_backward(dldf2, a2, f2); % ReLu back-propagation for second layer = dl/df2 * df2/da2
        
        [dldf1, dldw2, dldb2] = FC_backward(dlda2, f1, w2, b2, a2); % FC back-propagation
        %dldf1 = dl/da2 * da2/df1; dldw2 = dl/da2 * da2/dw2; dldb2 = dl/da2 * da2/db2
        
        % Update w2 and b2
        dLdw2 = dLdw2 + dldw2;
        dLdb2 = dLdb2 + dldb2;
        
        dlda1 = ReLu_backward(dldf1, a1, f1); % ReLu back-propagation for first layer = dl/df1 * df1/da1
        [dldx, dldw1, dldb1] = FC_backward(dlda1, x, w1, b1, a1); % FC back-propagation
        
        % Update w1 and b1
        dLdw1 = dLdw1 + dldw1;
        dLdb1 = dLdb1 + dldb1;
    end
    % Set k = 1 if k is greater than the number of mini-batches else k++
    if k >= length(mini_batch_x)
        k = 1;
    else
        k = k + 1;
    end
    
    % Update the weights and bias
    w2 = w2 - ((gamma/batch_size) * reshape(dLdw2,[size(dLdw2,2),size(dLdw2,3)]));
    b2 = b2 - ((gamma/batch_size) * dLdb2');
    
    w1 = w1 - ((gamma/batch_size) * reshape(dLdw1,[size(dLdw1,2),size(dLdw1,3)]));
    b1 = b1 - ((gamma/batch_size) * dLdb1');
end


