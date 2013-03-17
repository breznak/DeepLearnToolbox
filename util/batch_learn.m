function [nn, L]  = batch_learn(spec_sniplet, nn, train_x, train_y, opts, val_x, val_y)
%% NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 5 || nargin == 7,'number of input arguments must be 5 or 7')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];

opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if nn.plot == 1
    fhandle = figure();
end

m = size(train_y, 2);

batchsize = nn.batchsize;
numepochs = nn.numepochs;
numbatches = m / batchsize;

assert(rem(numbatches,1)==0, 'numbatches must be an integer');

L = zeros(numepochs*numbatches,1);
nn.rL=[]; % running loss
n = 1;
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        if strcmp(nn.type,'cnn')
            batch_x = train_x(:, :, kk((l - 1) * batchsize + 1 : l * batchsize));
            batch_y = train_y(:,    kk((l - 1) * batchsize + 1 : l * batchsize));
        else
            batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
            batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        end
        nn = feval(spec_sniplet,nn,batch_x,batch_y); % call specific Fn
        
        L(n) = nn.L; % MSE for each batch
        n = n + 1;
        % running average error
        if isempty(nn.rL)
                nn.rL(1) = nn.L;
        end
        nn.rL(end + 1) = 0.99 * nn.rL(end) + 0.01 * nn.L;
    end
    
    t = toc;
    
    if nn.plot == 1
        if opts.validation == 1
            loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        else
            loss = nneval(nn, loss, train_x, train_y);
        end
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
    disp(['epoch ' num2str(i) '/' num2str(nn.numepochs) '. Took ' num2str(t) ' seconds' '. Mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1))))]);
    nn.alpha = nn.alpha * nn.scaling_learningRate;
end
end

