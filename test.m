clear all;

n = nn([1000,50,200,11],{'sigm','tanh','tanh','smax'});
n.trainer = 'newton';
n.options.batchSize = 100;
n.options.learningRate = 0.01;
n.options.hessianStep = .00001;
n.options.epochs = 1;
n.options.visual = false;
n.options.dropoutProb = 0;

x = [-10:1:10];
N = numel(x);
d = x.*sin(x);
t = triu(bsxfun(@min,d,d.'),1); % The upper trianglar random values
M = diag(d)+t+t.'; % Put them together in a symmetric matrix
Mvec = M(:);

n.X = sin(rand(10000, numel(Mvec))*4 + repmat(Mvec, 1, 10000)' * 5 )/5;
n.Y = rand(10000,11)/5 + ( repmat([-1:.2:1],10000,1).^2 );

n.Xval = sin(rand(100, numel(Mvec))*4 + repmat(Mvec, 1, 100)' * 5 )/5;
n.Yval = rand(100,11)/5 + ( repmat([-1:.2:1],100,1).^2 );

n.train;

return

%% MNIST Prep data
load('C:\Users\dnadler\Downloads\Orig.mat')
for i = 1:numel(gnd)
    label(i,gnd(i)+1) = 1;
end
trainX = fea(1:60000,:);
trainY = label(1:60000,:);
testX = fea(60001:end,:);
testY = label(60001:end,:);
clear gnd fea label;

%% MNIST test

n = nn( [1000, 100, 10], {'sigm','tanh','smax'} );
n.trainer = 'newton';
n.options.epochs = 5;
n.options.visual = false;
n.options.learningRate = 0.01;
n.options.dropoutProb = 0.5;
n.options.batchSize = 1000;

n.X = trainX;
n.Y = trainY;

n.Xval = testX;
n.Yval = testY;

n.train;