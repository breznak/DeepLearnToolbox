function nn = sniplet_cnn(nn,batch_x,batch_y)
%% sniplet for cnn specific learning
% called from nnlearn->batch_learn
        nn = cnnff(nn, batch_x);
        nn = cnnbp(nn, batch_y);
        nn = cnnapplygrads(nn);
end