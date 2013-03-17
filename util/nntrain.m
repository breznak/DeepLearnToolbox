function [nn, L] = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%% NNTRAIN train any network on given data, (optional validation check)
% nntrain is polymorphic fn, calls specific *nn function: 
% nntrain -> batch_learn -> specific sniplet
switch nn.type
    case 'ffnn'
        %% NNTRAIN trains a neural net
        % [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
        % output y for opts.numepochs epochs, with minibatches of size
        % opts.batchsize. Returns a neural network nn with updated activations,
        % errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
        % squared error for each training minibatch.
        [nn,L]=batch_learn(@sniplet_ffnn,nn, train_x, train_y, opts, val_x, val_y);
    case 'cnn'
        %% CNNTRAIN train CNN - convolutional neural net, 
        % esp. good for image recognition
        % x
        % y
        [nn,L]=batch_learn(@sniplet_cnn,nn, train_x, train_y, opts, val_x, val_y);
    case 'sae'
        [nn,L]=batch_learn(@sniplet_sae,nn, train_x, train_y, opts, val_x, val_y);
    case 'dbn' 
        [nn,L]=batch_learn(@dbntrain,nn, train_x, train_y, opts, val_x, val_y);
    otherwise
        error('nntrain: unknown nn.type');
end
end