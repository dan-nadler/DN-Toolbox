classdef cnn < nn & handle

    properties( Access = public )
        convLayers
        Wc
        Bc
        Fc
    end

    methods
    
        function obj = cnn( conv_layers, layer_size, layer_type )
            
            obj = obj@nn( layer_size, layer_type );
            obj.trainer = 'cnn';
            obj.convLayers = conv_layers;
            
            % Conv Nets get very large. These are turned off to conserve memory.
            obj.options.log = true;
            obj.options.revertToBest = false;
            
            % set conv layer activation functions
            for i = 1:numel(conv_layers)
                try
                    obj.Fc{i,1} = obj.layers.(conv_layers{i}.aFxn).fxn;
                    obj.Fc{i,2} = obj.layers.(conv_layers{i}.aFxn).grad;
                catch ME
                    msg = [ 'Did not recognize layer type: ' conv_layers{i}.aFxn ];
                    causeException = MException( 'dsnn:setF', msg );
                    ME = addCause( ME, causeException );
                    throw(ME);
                end
            end
            
        end
        
        function obj = train( obj )
            
            if isempty( obj.W ) || isempty( obj.Wc )
                
                if obj.options.verbose
                    fprintf('Initializing model with random values.\n\n');
                end
                
                % initialize weights and biases with random values
                obj.randomInit();
                
            end
            
            if obj.options.log == true
                obj.logs.Wc{1} = obj.Wc{1};
                obj.logs.Bc{1} = obj.Bc{1};
            end
            
            obj = train@nn( obj );
        end
        
        function output = propOutputFromInput( obj, input )
            
            in{1} = input;
            
            for i = 1:numel( obj.convLayers )
                in{i+1} = obj.convolve( in{i}, obj.Wc{i}, obj.Bc{i}, obj.convLayers{i}.pSize );
            end
            
            propIn = reshape( in{end}, size(in{end}, 1), ...
                size( in{end}, 2 ) * size( in{end}, 3 ) * size( in{end}, 4 ) );
            
            output = propOutputFromInput@nn( obj, propIn );
        end
        
    end
    
    methods ( Static )
        [ output, A ] = convolve( input, W, B, pSize )
        % convolve( input, weights, kSize, pSize )
        
        [ input, dW ] = provolve( err, weights, kSize, pSize, pts, dA )
        % provolve( err, weights, kSize, pSize )
    end
    
    methods ( Access = protected )
        function randomInit( obj )
            
            for i = 1:numel(obj.convLayers)
                nChs = size(obj.X,3);
                kSize = obj.convLayers{i}.kSize;
                fNum = obj.convLayers{i}.nFeature;

                obj.Wc{i} = randn( fNum, kSize );
                obj.Bc{i} = randn( fNum, 1 );
            end
            
            randomInit@nn( obj, obj.convLayers{end}.pSize * nChs * fNum );
            
        end
    end

end