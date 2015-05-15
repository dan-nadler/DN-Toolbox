clear all;

noise = 5;

n = 10000;
x = rand(n,3).*2-1;
b = rand(n,3).*6;
e = rand(n,1).*2-1;

%y = x(:,1).*-1 + x(:,2).*2 + (x(:,1).^2).*2  + (x(:,2).^2).*-2 + e.*0;
%y = x(:,1).*-1 + x(:,2).*2 + (x(:,1).^2).*2;
% y = x(:,1).*-1 + x(:,2).*2 + (x(:,1).^2).*2 + (x(:,1).*x(:,2)).*1;
y = sum( x .* b>3, 2)  + sum( ( x .* b<3 ) .^ 2 , 2);

y = y + e.*noise;
y = ((y-min(y))./(max(y)-min(y)));
%y = ((y-min(y))./(max(y)-min(y))).*2-1;

y_temp = y;

all = (1:n);
is = randperm(n,n.*.75);
os = setdiff(all,is);

model = ols_train(x(is,:),y(is));
yfit_ols = ols_predict(model,x(is,:));
yhat_ols = ols_predict(model,x(os,:));

% xx = [-x(:,1) 2*x(:,2) (x(:,1).^2).*2 x(:,1).*x(:,2)];
xx = [x .* b>3 (x .* b<3 ) .^ 2];
model = ols_train(xx(is,:),y(is));
yfit_perf = ols_predict(model,xx(is,:));
yhat_perf = ols_predict(model,xx(os,:));

denoiser = n.


clear n
%n = nn([30, 20, 10, 5, 1],{'relu','relu','relu','relu','relu'});
%n = nn([100, 50, 5, 1],{'relu','relu','relu','relu'});
n = nn([5,4,3,2,1],{'tanh','tanh','tanh','tanh','tanh'});
%n = nn([10,1],{'relu','relu'});
% n.trainer = 'backprop';
n.trainer = 'newton';
n.options.batchSize = 10;
n.options.learningRate = 0.05;
n.options.hessianStep = .00001;
n.options.epochs = 10;
n.options.visual = false;
n.options.dropoutProb = 0;
n.options.verbose = true;
n.options.revertToBest = true;
n.options.corr = true;

n.X = x(is,:);
n.Y = y(is,:);

n.Xval = x(os,:);
n.Yval = y(os,:);

n.train;

% n.options.learningRate = 0.005;
% 
% n.train;

yfit_nn = n.propOutputFromInput(n.X);
yhat_nn = n.propOutputFromInput(n.Xval);


disp(' ');
disp('best possible model')
disp([nancorr(yfit_perf,y(is)) nancorr(yhat_perf,y(os))]);
disp('linear model')
disp([nancorr(yfit_ols,y(is)) nancorr(yhat_ols,y(os))]);
disp('neural-net')
disp([nancorr(yfit_nn,y(is)) nancorr(yhat_nn,y(os))]);

% subplot(3,1,1); scatter(y(is),yfit_perf);
% subplot(3,1,2); scatter(y(is),yfit_ols);
% subplot(3,1,3); scatter(y(is),yfit_nn);
% shg