function obj = trainer_conv_net( obj )   
    
    obj.options.dropoutProb = 0;

    numLayers = size(obj.F,1);
    numConvLayers = size(obj.convLayers);
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
        A{1,1} = obj.X(s:e,:); % input layer
        input{1,1} = A{1,1};
        
        % for each conv-pool layer
        for ic = 1:numConvLayers
            % convolve
            % construct 3D matrix of convolution ops
            conv_mat = obj.Wc{1} .* ones(1, obj.convLayers.kSize, obj.convLayer.nFeature) / obj.convLayers.kSize;
            
            % for each feature map
            for ifea = 1:size(conv_vec,3)
                conv_vec = conv_mat(1, :, ifea);
                for ib = 1:actBatchSize
                    % perform convolve op
                    conv_output(ib,:,ifea) = conv2( input{1}(ib,:), conv_vec, 'same' );
                    % trim output vector
                    conv_output(ib,:,ifea) = output(1:end-1);
                end
            end
        end
        
        for i = 1:numLayers

            % hidden and output layer activation
            input{i+1,1} = A{i} * obj.W{i} + repmat(obj.B{i},actBatchSize,1); %input to layer
            drop{i+1,1} = dropout( input{i+1} );
            A{i+1,1} = obj.F{i,1}( input{i+1,1} ) .* drop{i+1}; % activation of layer
            dA{i+1,1} = obj.F{i,2}( input{i+1,1} ) .* drop{i+1}; % activation gradient of layer
            
        end
        
        
    end

    
end