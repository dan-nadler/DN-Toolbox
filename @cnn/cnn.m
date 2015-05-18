classdef cnn < nn & handle

    properties( Access = public )
        convLayers
        Wc
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
        
    end
    
    methods ( Access = protected )
        function randomInit( obj )
            
            inputSize = size( obj.X, 2 );
            obj.Wc{1} = randn( 1, inputSize(2) - obj.convLayers{1}.kSize + 1, obj.convLayers{1}.nFeature );
            
            for i = 2:numel( obj.convLayers )
                obj.Wc{i} = randn( 1, obj.convLayers{i-1}.pSize + 1, obj.convLayers{1}.nFeature );
            end
            
            randomInit@nn( obj, numel(obj.Wc{end}) );
            
        end
    end

end