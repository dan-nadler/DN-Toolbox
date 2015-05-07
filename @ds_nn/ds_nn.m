classdef ds_nn < handle
% Neural Net class
% author: Dan Nadler

    properties ( Access = public )
        X % input data
        Y % output (target) data
        N % layer size
        F % layer type / activation function
        dF % layer gradient
        B % biases
        W % weights
        trainer % eg. backprop
        layers % all supported layer types
        regs % all supported regularizers
        options % various options (batch size, learning rate, etc.)
        logs % stores snapshots of model performance during training
    end
    
    properties ( Hidden = true, Access = public )
        
    end
    
    
    methods
        function obj = ds_nn( layer_size, layer_type )
        % ds_nn( layer_sizes, layer_types )
        % layer_sizes = cell array of layer sizes
        % layer_types = cell array of layer types
            
            % set default options
            obj.options.batchSize = 100;
            obj.options.learningRate = .15;
            obj.options.hessianStep = .3;
            obj.options.epochs = 30;
            obj.options.visual = false;
            % obj.options.descent = 'Newton'; % 'SGD', 'SFN', 'Newton'
            
            obj.logs.rmse = [];
            obj.logs.W = {};
            obj.logs.hessian = {};
%             obj.logs.learnMovie = struct;
        
            % store supported layers
            obj.layers.sigm.fxn = @(x) 1 ./ (1 + exp(-x) ); %sigmoid            
            obj.layers.sigm.grad = @(x) ( 1 ./ (1 + exp(-x) ) ) .* ( 1 - (1 ./ (1 + exp(-x) )) );
            
            obj.layers.tanh.fxn = @(x) tanh(x); %hyperbolic tangent
            obj.layers.tanh.grad = @(x) sech(x).^2;
            
            obj.layers.smax.fxn = @(x) exp(x) ./ repmat(sum(exp(x),2),1,size(x,2)); %softmax
            obj.layers.smax.grad = @(x) nan;
            
            obj.layers.relu.fxn = @(x) max( x, 0 ); %rectified linear
            obj.layers.relu.grad = @(x) x > 0;
            
            % store supported regularizers
            obj.regs.fxn.l2 = @(x) sqrt(sum(x.^2,2))*2;
            obj.regs.fxn.none = @(x) 0;
            obj.regs.active = obj.regs.fxn.l2;
            
            % set model size and type
            assert( numel(layer_size) == numel(layer_type),...
                'Layer size cell array is not the same size as layer functions array ( numel(N) ~= numel(F) )');
            obj.N = layer_size;
            obj.F = layer_type;
            
            % set data
            obj.X = []; 
            obj.Y = []; 
            
            obj.trainer = 'newton'; % backprop
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
                error('DeepStocks:SetN','Could not read N. Type should be Array or Cell Array');
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
                    error('DeepStocks:SetF','Could not read F. Type should be Cell Array');
                end
            end
        end
        
        function obj = set.trainer( obj, trainer )
        % set trainer
            switch trainer
                case 'backprop'
                    obj.trainer = @trainer_backprop;
                case 'newton'
                    obj.trainer = @trainer_backprop_quasi_newton;
            end
        end
        
        function obj = train( obj )
            
            if isempty( obj.W )
                %random weights and bias matrix initialization
                numLayers = size(obj.F,1);
                obj.W{1,1} = (rand(size(obj.X,2),obj.N{1})-.5)*2; %input to hidden weights matrix
                obj.B{1,1} = (rand(1,obj.N{1})-.5)*2; %1st hidden layer bias
                for i = 2:numLayers
                    obj.W{i,1} = (rand(size(obj.W{i-1},2),obj.N{i})-.5)*2; %weights matrix
                    obj.B{i,1} = (randn(1,obj.N{i})-.5)*2; %bias
                end

                obj.logs.W{end+1} = obj.W;
            end
            
            if obj.options.visual == true
                figure;
            end
               
            for i = 1:obj.options.epochs
                fprintf('Epoch %i ',i);
                obj = obj.trainer( obj );
                fprintf(' RMSE: %f\n',obj.logs.rmse(end) );
            end
            
        end
        
        function output = predict( obj, input )
        % given inputs, predict output
        % in the future, this will depend on the type of net created
            output = obj.forwardPropFxn( input );
        end
        
    end
    
    methods ( Access = protected )
       
        obj = trainer_backprop( obj );
        
        function activation = forwardPropFxn( obj, input )
        % construct forward propagate function
            
            activation = obj.F{1,1}( input * obj.W{1} + obj.B{1} );
            for i = 2:size( obj.F, 1 )
                activation = obj.F{i,1}( activation * obj.W{i} + obj.B{i} );
            end
            
        end
        
    end
    
    methods ( Static )
        
    end
end