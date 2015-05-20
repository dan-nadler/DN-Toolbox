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
        
        function testInit( obj )
            randomInit(obj);
        end
        
    end
    
    methods ( Static )
        [ output ] = convolve( input, W, B, pSize )
        % convolve( input, weights, kSize, pSize )
        
        [ input, dW ] = provolve( err, weights, kSize, pSize, pts, dA )
        % provolve( err, weights, kSize, pSize )
    end
    
    methods ( Access = protected )
        function randomInit( obj )
            
            for i = 1:numel(obj.convLayers)
                nChs = size(obj.X,3);
                kSize = obj.convLayers{i}.kSize;
                fNum = obj.obj.convLayers{i}.nFeature;

                obj.Wc{i} = randn( fNum, kSize );
                obj.Bc{i} = randn( fNum, 1 );
            end
            
            randomInit@nn( obj, obj.convLayers{end}.pSize * size(obj.X,3) );
            
        end
    end

end