function obj = trainer_conv_net( obj )   
    
    obj.options.dropoutProb = 0;

    numLayers = size(obj.F,1);
    loss = obj.loss;
    reg = obj.regs.active;
    batchSize = obj.options.batchSize;
    dropout = @(mat) rand(size(mat)) > obj.options.dropoutProb;
    layerTypes = obj.options.iniParams.layer_type;
    
    b = 0;
    while batchSize*b < size(obj.X,1)-1
        b = b + 1;
        e = min( batchSize * b, size(obj.X,1) );
        s = batchSize * b - batchSize + 1;
        actBatchSize = e-s+1;

        % forward prop
        % input layer activation
        drop{1,1} = dropout(obj.X(s:e,:));
        A{1,1} = obj.X(s:e,:) .* drop{1,1}; % input layer
        input{1,1} = A{1,1};
        
        for i = 1:numLayers
            switch layerTypes{i}
                
                case 'conv'
                    convVec{i,1} = repmat( 1/obj.N{i}, 1, obj.N{i} );
                    for j = 1:actBatchSize
                        temp = conv( convVec{i,1}, input{i}(j,:) );
                        input{i+1,1}(j,:) = temp( obj.N{i} : end-obj.N{i}+1 );
                    end
                    
                case 'meanpool'
                    pSize{i,1} = ceil( size( input{i}, 2 ) / (obj.N{i}) );
                    for j = 1:actBatchSize
                        input{i+1,1}(j,:) = ...
                            obj.F{i,1}( reshape( input{i}(j,:), pSize{i}, numel(input{i})/pSize{i} ) );
                    end
                    
                otherwise
                    % hidden and output layer activation
                    input{i+1,1} = A{i} * obj.W{i} + repmat(obj.B{i},actBatchSize,1); %input to layer
                    drop{i+1,1} = dropout( input{i+1} );
                    A{i+1,1} = obj.F{i,1}( input{i+1,1} ) .* drop{i+1}; % activation of layer
                    dA{i+1,1} = obj.F{i,2}( input{i+1,1} ) .* drop{i+1}; % activation gradient of layer
                    
            end
            
        end
        
        
    end

    
end