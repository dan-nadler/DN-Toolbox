% addpath('C:\Users\dnadler\machine_learning\DeepStocks');

clear all;
n = ds_nn([50,100,20, 11],{'sigm','tanh','tanh','tanh'});
n.trainer = 'hessian';
n.options.batchSize = 5;
n.options.learningRate = 0.05;
n.options.hessianStep = .01;
n.options.epochs = 10;
n.options.visual = true;
n.X = sin(rand(10000,11)*4 + repmat([-1:.2:1],10000,1) * 5 )/5;
n.Y = rand(10000,11)/5 + ( repmat([-1:.2:1],10000,1).^2 );
n.train;
% plot(n.predict([-1:.2:1]),'color','red');