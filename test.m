clear all;

n = ds_nn([1000,50,200,11],{'sigm','tanh','tanh','tanh'});
n.trainer = 'newton';
n.options.batchSize = 100;
n.options.learningRate = 0.01;
n.options.hessianStep = .00001;
n.options.epochs = 10;
n.options.visual = true;
n.options.dropoutProb = 0.1;

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