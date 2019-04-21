function [w_conv, b_conv, w_fc, b_fc] = TrainCNN(mini_batch_x, mini_batch_y)
%Output: w_conv ? R3󫢩�3, b_conv ? R3, w_fc ? R10�147, b_fc ? R147 are the trained weights
%and biases of the CNN.
%Description: You will use the following functions to train a convolutional neural network
%using a stochastic gradient descent method: Conv, Conv_backward, Pool2x2, Pool2x2_backward,
%Flattening, Flattening_backward, FC, FC_backward, ReLu, ReLu_backward, Loss_cross_entropy_softmax.
%As a result of training, the network should produce more than 92% of accuracy on the testing data.

m = size(mini_batch_x{1},1); % size of each image which is 196
n = size(mini_batch_y{1},1); % size of each label (one-hot encoding) which is 10

% Set the learning rate and the decay rate ? (0, 1]
gamma = 0.01;
lambda = 0.6;

% Initialize the weights with a Gaussian noise w ? N (0, 1) and bias
w_conv(:,:,1) = [1 -1 1;0 1 0;-1 1 0];
w_conv(:,:,2) = [-1 1 0;1 1 -1;0 1 -1];
w_conv(:,:,3) = [-1 0 0;0 1 1;0 1 1];
b_conv = rand([3,1]);
s = (sqrt(m)/2).^2 * length(b_conv);
w_fc = rand([n,s])*sqrt(2/(n+m));
b_fc = rand([n,1])*sqrt(2/(n));

k = 1; % initialize k=1
nIters = 10000;
for iter = 1:nIters
    iter
    if mod(iter,1000) == 0 %6: at every 1000th iteration, ? ? ??
        gamma = lambda*gamma;
    end
    
    % Set dL/dw and dL/db to 0
    dLdw_fc = 0; dLdw_conv = 0;
    dLdb_fc = 0; dLdb_conv = 0;
    
    % Loop for each image xi in kth mini-batch
    batch_size = size(mini_batch_x{k},2); % num of images in kth mini_batch
    for i = 1:batch_size
        x = reshape(mini_batch_x{k}(:,i), [14, 14, 1]); % each image in kth mini_batch
        y = mini_batch_y{k}(:,i); % ground truth label
        k
        i
        % Forward
        pred1 = Conv(x, w_conv, b_conv); % Conv layer
        pred2 = ReLu(pred1); % ReLu
        pred3 = Pool2x2(pred2); % Max Pooling
        pred4 = Flattening(pred3); % Flattening
        pred5 = FC(pred4, w_fc, b_fc); % FC
        
        [l, dldy] = Loss_cross_entropy_softmax(pred5, y); % softmax and cross entropy loss
        
        % Back propagation
        [dldx, dldw_fc, dldb_fc] = FC_backward(dldy, pred4, w_fc, b_fc, pred5); % FC-back
        dldf = Flattening_backward(dldx, pred3, pred4); % Flattening-back
        dldp = Pool2x2_backward(dldf, pred2, pred3); % Pooling-back
        dldr = ReLu_backward(dldp, pred1, pred2); % ReLu-back
        [dldw_conv, dldb_conv] = Conv_backward(dldr, x, w_conv, b_conv, pred1); % Conv-back
        
        % Update dLdw and dLdb
        dLdw_fc = dLdw_fc + dldw_fc;
        dLdb_fc = dLdb_fc + dldb_fc;
        dLdw_conv = dLdw_conv + dldw_conv;
        dLdb_conv = dLdb_conv + dldb_conv;
    end
    % Set k = 1 if k is greater than the number of mini-batches else k++
    if k >= length(mini_batch_x)
        k = 1;
    else
        k = k + 1;
    end
    
    % Update the weights and bias
    w_conv = w_conv - ((gamma/batch_size) * reshape(dLdw_conv,[size(dLdw_conv,2),size(dLdw_conv,3)]));
    b_conv = b_conv - ((gamma/batch_size) * dLdb_conv');
    
    w_fc = w_fc - ((gamma/batch_size) * reshape(dLdw_fc,[size(dLdw_fc,2),size(dLdw_fc,3)]));
    b_fc = b_fc - ((gamma/batch_size) * dLdb_fc');
end



