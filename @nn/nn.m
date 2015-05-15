classdef nn < handle
% Neural Net class
% author: Dan Nadler
%
% Functions
% 
% nn( layer_size <cell array>, layer_type <cell array> )  
%
% train()
%
% output <vector> = propOutputFromInput( input <vector> )
%
% input <vector> = propInputFromOutput( ouput <vetor> )
%
% RMSE <float> = calcRMSE( input <vector>, output <vector> )
%

    properties ( Access = public )
        X % input data
        Y % output (target) data
        Xval % input validation sample
        Yval % output validation sample
        N % layer size
        F % layer type / activation function
        B % biases
        W % weights
        trainer % eg. backprop
        layers % all supported layer types
        regs % all supported regularizers
        options % various options (batch size, learning rate, etc.)
        logs % stores snapshots of model performance during training
        loss % loss function
    end
    
    properties ( Hidden = true, Access = public )
        
    end
    
    methods
        function obj = nn( layer_size, layer_type )
        % ds_nn( layer_sizes, layer_types )
        % layer_sizes = cell array of layer sizes
        % layer_types = cell array of layer types
            
            % set default options
            obj.options.batchSize = 100;
            obj.options.learningRate = .05;
            obj.options.hessianStep = .0001;
            obj.options.epochs = 30;
            obj.options.dropoutProb = 0;
            obj.options.visual = false;
            obj.options.verbose = false;
            obj.options.corr = false;
            obj.options.revertToBest = true;
            obj.options.gpu = false;
            
            obj.options.iniParams.layer_size = layer_size;
            obj.options.iniParams.layer_type = layer_type;
            
            obj.logs.rmse = [];
            obj.logs.rmse_batch = [];
            obj.logs.rmse_val = [];
            obj.logs.corr = [];
            obj.logs.corr_val = [];
            obj.logs.W = {};
            obj.logs.B = {};
            obj.logs.hessian = {};
            % obj.logs.learnMovie = struct;
        
            % store supported layers
            obj.layers.sigm.fxn = @(x) 1 ./ (1 + exp(-x) ); %sigmoid            
            obj.layers.sigm.grad = @(x) ( 1 ./ (1 + exp(-x) ) ) .* ( 1 - (1 ./ (1 + exp(-x) )) );
            
            obj.layers.tanh.fxn = @(x) tanh(x); %hyperbolic tangent
            obj.layers.tanh.grad = @(x) sech(x).^2;
            
            obj.layers.hardtanh.fxn = @(x) max(-1,min(1,tanh(x))); % hard tanh
            obj.layers.hardtanh.grad = @(x) ( x>-1 & x<1 ) .* ( sech(x) .^ 2 );
            
            obj.layers.smax.fxn = @(x) exp(x) ./ repmat(sum(exp(x),2),1,size(x,2)); %softmax
            obj.layers.smax.grad = @(x) zeros(size(x)); % this is a problem... need to figure out gradient if smax is ever going to be a hidden layer
            
            obj.layers.relu.fxn = @(x) max( x, 0 ); %rectified linear
            obj.layers.relu.grad = @(x) x > 0;
            
            obj.layers.avrect.fxn = @(x) abs(x); % absolute value
            obj.layers.avrect.grad = @(x) sign(x);
            
            % store supported regularizers
            obj.regs.fxn.l2 = @(x) sqrt(sum(x.^2,2))*2;
            obj.regs.fxn.none = @(x) 0;
            obj.regs.active = obj.regs.fxn.l2;
            
            % set model size and type
            assert( numel(layer_size) == numel(layer_type),...
                'Layer size cell array is not the same size as layer functions cell array ( numel(N) ~= numel(F) )');
            obj.N = layer_size;
            obj.F = layer_type;
            
            % set loss function
            obj.loss = '';
            
            % set data
            obj.X = []; 
            obj.Y = [];
            
            obj.Xval = [];
            obj.Yval = [];
            
            obj.trainer = 'newton'; % default trainer is quasi-newtorn
        end
        
        function obj = set.N( obj, N )
        % set layer size
            if iscell(N)
                obj.N = N;
            elseif ismatrix(N)
                Nc = cell(numel(N),1);
                for i = 1:numel(N)
                    Nc{i,1} = N(i);
                    obj.N = Nc;
                end
            else
                error('PAIL:Set:N','Could not read N. Type should be Array or Cell Array');
            end
        end
        
        function obj = set.F( obj, Fxns )
        % set layer types by matching provided strings with available types in obj.layers struct
            for i = 1:numel(Fxns)
                if iscell(Fxns)
                    try
                        obj.F{i,1} = obj.layers.(Fxns{i}).fxn;
                        obj.F{i,2} = obj.layers.(Fxns{i}).grad;
                    catch ME
                        msg = [ 'Did not recognize layer type: ' Fxns{i} ];
                        causeException = MException( 'dsnn:setF', msg );
                        ME = addCause( ME, causeException );
                        throw(ME);
                    end                        
                else
                    error('PAIL:Set:F','Could not read F. Type should be Cell Array');
                end
            end
        end
        
        function obj = set.trainer( obj, trainer )
        % set trainer
            switch trainer
                case 'sgd'
                    obj.trainer = @trainer_backprop_quasi_newton;
                case 'backprop'
                    obj.trainer = @trainer_backprop_quasi_newton;
                case 'newton'
                    obj.trainer = @trainer_backprop_quasi_newton;
                case 'newtonGPU'
                    obj.trainer = @trainer_backprop_quasi_newton_gpu;
                case 'svm'
                    obj.trainer = @trainer_tied_weights;
                otherwise
                    error('PAIL:Set:Trainer',['Invalid trainer requested: ' trainer]);
            end
        end
        
        function obj = set.loss( obj, lossFxn )
            switch lossFxn
                case 'squared'
                    obj.loss = @(yh,y) ((yh-y).^2)/-2;
                otherwise
                    if strcmp(obj.options.iniParams.layer_type{end}, 'smax')
                        obj.loss = @(yh,y) ((yh-y).^2)/-2;
                    else
                        obj.loss = @(yh,y) (yh-y);
                    end
            end
        end
        
        function obj = train( obj )
            
            if isempty( obj.W )
                
                if obj.options.verbose
                    fprintf('Initializing model with random values.\n\n');
                end
                
                % initialize weights and biases with random values
                obj.randomInit();
                
            end
            
            if obj.options.visual == true
                figure;
            end
               
            for i = 1:obj.options.epochs
                
                if obj.options.verbose
                    % list current epoch
                    fprintf( 'Epoch %i:\t\t', i );

                    % run training algorithm for 1 epoch
                    obj = obj.trainer( obj );

                    % RMSE of in-sample data
                    model_Y = obj.propOutputFromInput( obj.X );
                    rmse = obj.calcRMSE( model_Y, obj.Y );
                    fprintf( 'RMSE: %f\t', rmse );
                    obj.logs.rmse(end+1,1) = rmse;
                
                    % RMSE of validation data
                    if ~isempty( obj.Yval )
                        model_Yval = obj.propOutputFromInput( obj.Xval );
                        rmse_val = obj.calcRMSE( model_Yval , obj.Yval);
                        fprintf( 'Validation RMSE: %f\t', rmse_val );
                        obj.logs.rmse_val(end+1,1) = rmse_val;
                    end
                    
                    % Correlation
                    if obj.options.corr
                        
                        % in sample
                        corr = nancorr( obj.Y, model_Y );
                        fprintf( 'Corr: %f\t', corr );
                        obj.logs.corr(end+1,1) = corr;
                        
                        % validation data
                        if ~isempty( obj.Yval )
                            corr_val = nancorr( obj.Yval, model_Yval );
                            fprintf( 'Val Corr: %f\t', corr_val );
                            obj.logs.corr_val(end+1,1) = corr_val;
                        end
                        
                    end
                    
                    fprintf('\n');
                    
                end
                
            end
            
            if obj.options.verbose
                fprintf('\n');
            end
            
            % Revert model to lowest error / highest correl
            if obj.options.revertToBest
                if obj.options.corr
                    if ~isempty( obj.Yval )
                        best_epoch = find( obj.logs.corr_val == max( obj.logs.corr_val ) );
                    else
                        best_epoch = find( obj.logs.corr == max( obj.logs.corr ) );
                    end
                else
                    if ~isempty( obj.Yval )
                        best_epoch = find( obj.logs.rmse_val  == min( obj.logs.rmse_val ) );
                    else
                        best_epoch = find( obj.logs.rmse  == min( obj.logs.rmse ) );
                    end
                end
            
                if best_epoch < numel( obj.logs.rmse )
                    fprintf( 'Reverting weights and biases to epoch %i.\n', best_epoch );
                    obj.W = obj.logs.W{best_epoch};
                    obj.B = obj.logs.B{best_epoch};
                end
            end
            
        end
        
        function output = propOutputFromInput( obj, input )
        % given inputs, predict output
        % in the future, this will depend on the type of net created
            output = obj.propOutputFromInputFxn( input );
        end
        
        function activation = propInputFromOutput( obj, output )
        % construct function to activate input layer, given output
            temp = obj.propInputFromOutputFxn( output );
            activation = temp{1};
        end
        
    end
    
    methods ( Static )
        
        function output = calcRMSE( y, y_hat )
        % calculate the RMSE given Y and Y hat
            output = sqrt( mean( sum( (y - y_hat) .^ 2, 1 ) / size(y_hat,1) ) );
        end
        
    end
    
    methods ( Access = protected )
       
        obj = trainer_backprop( obj );
        
        function activation = propOutputFromInputFxn( obj, input )
        % construct forward propagate function
            
            activation = input;
%             activation = obj.F{1,1}( input * obj.W{1} + repmat( obj.B{1}, size(input,1), 1 ) );
            for i = 1:size( obj.F, 1 )
                activation = obj.F{i,1}( activation * obj.W{i} + repmat( obj.B{i}, size(input,1), 1 ));
            end
            
        end
        
        function activation = propInputFromOutputFxn( obj, output )
        % construct function to activate input layer, given output
            activation{numel(obj.W)+1,1} = output;
            b = cell( numel( obj.B ) + 1, 1 );
            b(2:end) = obj.B;
            b(1) = { zeros( 1, size( obj.X,2 ) ) };
            for i = numel(obj.W):-1:1
                activation{i,1} = activation{i+1} * obj.W{i}' + repmat( b{i}, size(output,1), 1 );
            end
        end
        
        function randomInit( obj )
        %random weights and bias matrix initialization
            numLayers = size(obj.F,1);
            obj.W{1,1} = randn(size(obj.X,2),obj.N{1})/3; %input to hidden weights matrix
            obj.B{1,1} = randn(1,obj.N{1})/3; %1st hidden layer bias
            for i = 2:numLayers
                obj.W{i,1} = (rand(size(obj.W{i-1},2),obj.N{i})-.5)*2; %weights matrix
                obj.B{i,1} = (randn(1,obj.N{i})-.5)*2; %bias
            end
        end
        
    end
    
    methods ( Static )
        
    end
    
end