clear all;

n = nn([11,20,11],{'hardtanh','hardtanh','hardtanh'});
n.trainer = 'newton';
n.options.batchSize = 10;
n.options.learningRate = 0.05;
n.options.hessianStep = .00001;
n.options.epochs = 1;
n.options.visual = true;
n.options.dropoutProb = 0;
n.options.verbose = true;

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

%% CNN test
clear
noise = 10;
n = 100000;
k = 10;
data(:,:,1) = reshape( ones(1,n) .* randn(1,n) , [n/k, k, 1] );
data(:,:,2) = reshape( ones(1,n) .* randn(1,n) , [n/k, k, 1] );
target = ( mean( data(:,1:3,1), 2 ) + mean( data(:,4:7,2), 2 ) ) / 2;

% add noise to X
data = data + ( randn(size(data))./noise );

h = size(data,1) *.9;

layer = convLayer( 5, 3, 'mean', 'sigm', 2, 'mean' );
t = cnn({layer}, {10,1},{'tanh','tanh'});
t.X = data(1:h,:,:);
t.Y = target(1:h,:);

t.Xval = data(h:end,:,:);
t.Yval = target(h:end,:);

t.options.epochs = 1;
t.options.batchSize = 100;
t.options.learningRate = 0.1;

t.train;

nn = nn( {60, 30, 1}, {'tanh','tanh','tanh'} );
nn.options.epochs = 15;
nn.options.learningRate = 0.1;
nn.options.batchSize = 100;

nn.X = [ data(1:h, :, 1), data(1:h, :, 2) ];
nn.Y = target(1:h,:,:);

nn.Xval = [ data(h:end, :, 1), data(h:end, :, 2) ];
nn.Yval = target(h:end,:,:);

nn.train;