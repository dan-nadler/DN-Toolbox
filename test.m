clear all;

n = ds_nn([100,50,20,11],{'sigm','tanh','tanh','tanh'});
n.trainer = 'newton';
n.options.batchSize = 1;
n.options.learningRate = 0.01;
n.options.hessianStep = .00001;
n.options.epochs = 2;
n.options.visual = true;

x = [-10:1:10];
N = numel(x);
d = x.*sin(x);
t = triu(bsxfun(@min,d,d.'),1); % The upper trianglar random values
M = diag(d)+t+t.'; % Put them together in a symmetric matrix
Mvec = M(:);
% surf(M);

n.X = sin(rand(numel(Mvec),10000)*4 + repmat(Mvec,1,10000) * 5 )/5;
n.Y = rand(10000,11)/5 + ( repmat([-1:.2:1],10000,1).^2 );
n.train;
% plot(n.predict([-1:.2:1]),'color','red');