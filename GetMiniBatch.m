function [mini_batch_x, mini_batch_y] = GetMiniBatch(im_train, label_train, batch_size)
%Input: im_train and label_train are a set of images and labels, and batch_size is the size of
%the mini-batch for stochastic gradient descent.
%Output: mini_batch_x and mini_batch_y are cells that contain a set of batches (images and labels, respectively).
%Each batch of images is a matrix with size 196×batch_size, and each batch of labels is a matrix with size
%10×batch_size (one-hot encoding). Note that the number of images in the last batch may be smaller than batch_size.
%Description: You may randomly permute the the order of images when building the batch,
%and whole sets of mini_batch_* must span all training data.

training_size = length(label_train);
% shuffle im_train and label_train in same order
rnd = randperm(training_size);
im_train = im_train(:, rnd);
label_train = label_train(rnd);

% apply encoding to label_tain
one_hot_label = zeros(10,training_size);
for i = 1:training_size
    one_hot_label(label_train(i)+1,i) = 1;
end

% break into a set of batches
num_of_batch = ceil(training_size / batch_size);
leftover = mod(training_size, batch_size); % get the size of the last batch
for i = 1:num_of_batch
    start_point = ((i-1) * batch_size) + 1;
    end_point = i * batch_size;
    if i == num_of_batch % if the last batch is smaller than batch size, the end_point is different
        end_point = (start_point + leftover) - 1;
        mini_batch_x{i} = im_train(:,start_point:end_point);
        mini_batch_y{i} = one_hot_label(:,start_point:end_point);
    else
        mini_batch_x{i} = im_train(:,start_point:end_point);
        mini_batch_y{i} = one_hot_label(:,start_point:end_point);
    end
end



