classdef cnn < nn & handle

    properties( Access = public )
        convLayers
        Wc
        
    end

    methods
    
        function obj = cnn( conv_layers, layer_size, layer_type )
            
            obj = obj@nn( layer_size, layer_type );
            obj.trainer = 'cnn';
            obj.convLayers = conv_layers;

        end
        
    end
    
    methods ( Access = protected )
        function randomInit( obj )
            
            inputSize = size( obj.X, 2 );
            obj.Wc{1} = randn( 1, inputSize(2) - obj.convLayers{1}.kSize + 1, obj.convLayers{1}.nFeature );
            
            for i = 2:numel( obj.convLayers )
                obj.Wc{i} = randn( 1, obj.convLayers{i-1}.pSize + 1 );
            end
            
            randomInit@nn( obj );
            
        end
    end

end