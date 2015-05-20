function obj = trainer_conv_net( obj )   
    
    obj.options.dropoutProb = 0;

    numLayers = size(obj.F,1);
    numConvLayers = numel(obj.convLayers);
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
        c_input{1,1} = obj.X(s:e,:,:); % input layer
        
        % for each conv-pool layer
        for ic = 1:numConvLayers
            
            [ c_input{ic+1,1}, dAc{ic,1}, Ac{ic,1} ] = obj.convolve( c_input{ic}, obj.Wc{ic}, ...
                obj.convLayers{ic}.kSize, obj.convLayers{ic}.pSize );
            
        end
        
        input{1,1} = reshape( c_input{end}, actBatchSize, size(c_input{end},2) * size(c_input{end},3) );
        A{1,1} = input{1};
        
        for i = 1:numLayers

            % hidden and output layer activation
            input{i+1,1} = A{i} * obj.W{i} + repmat(obj.B{i},actBatchSize,1); %input to layer
            drop{i+1,1} = dropout( input{i+1} );
            A{i+1,1} = obj.F{i,1}( input{i+1,1} ) .* drop{i+1}; % activation of layer
            dA{i+1,1} = obj.F{i,2}( input{i+1,1} ) .* drop{i+1}; % activation gradient of layer
            
        end
        
        % back prop
        
        % normal backprop
        % output layer error
        errorOut{numLayers,1} = loss( A{numLayers+1}, obj.Y(s:e,:) ) .* drop{numLayers+1};
        hessian{numLayers,1} = ( ...
                                    obj.F{numLayers,2}( input{numLayers+1} + obj.options.hessianStep * errorOut{numLayers,1} ) ...
                                  - obj.F{numLayers,2}( input{numLayers+1} ) ...
                                )...
                                / obj.options.hessianStep;
        
        for i = numLayers:-1:2
            % hidden layer error
            % error = prior layer error * weight matrix .* dF/dA .* current layer dF/dA
            errorOut{i-1,1} = errorOut{i} * obj.W{i}' .* dA{i} .* obj.F{i-1,2}( input{i} ) .* drop{i};
            hessian{i-1,1} = ( ...
                                    obj.F{i-1,2}( input{i} + obj.options.hessianStep * errorOut{i-1,1} ) ...
                                  - obj.F{i-1,2}( input{i} ) ...
                                )...
                                / obj.options.hessianStep;
        end
        
        % conv net backprop
        
        c_error{numConvLayers,1} = errorOut{1} * obj.W{1}';

        err = c_error{numConvLayers};
        weights = obj.Wc{numConvLayers};
        kSize = obj.convLayers{numConvLayers}.kSize;
        pSize = obj.convLayers{numConvLayers}.pSize;
        pts = size(obj.X,2);
        
        chs = size( err, 2 ) / pSize;
        obs = size( err, 1 );

        pOut = reshape( err, [obs, pSize, chs] );
        kron_vec = ones(1, (pts-kSize+1)/pSize ) / ((pts-kSize+1)/ pSize );

        upsample = nan( size( dAc{numConvLayers} ) );
        numU = size( dAc{numConvLayers},2 );

        dW = zeros( size( weights ) );

        for ic = 1:chs % for each channel

            W = weights( ic, : );

            for io = 1:obs % for each observation
                
                % reverse the pooling operation
                % pSize -> pts - kSize + 1
                upsample( io, :, ic ) = kron( pOut(io,:,ic), kron_vec ) .* dAc{numConvLayers}( io, :, ic );

                for iu = 1:numU % for each convolution output

                    % calculate the dW for this channel's weight
                    dW( ic, :, io ) = dW( ic, :, io ) + obj.options.learningRate * ...
                        ( W * upsample(io,iu,ic) );
                    
                end
                
                dW( :, :, io ) = dW( :, :, io ) / numU;

            end

        end
    
        % update weights and biases
        
        for i = 1:numLayers
            % weight = old weight - learnRate * ( ( lambda * regularizer ) + ( activation * error ) ) 
            obj.W{i} = obj.W{i} - obj.options.learningRate * ...
                ( ( repmat( 0.01 * reg(obj.W{i}), 1, obj.N{i} ) .* sign(obj.W{i} ) ...
                + ( A{i}' * errorOut{i} ) + ( A{i}' * hessian{i} ) ) ) ...
                / actBatchSize;
            
            obj.B{i} = obj.B{i} - obj.options.learningRate * ( sum( errorOut{i}, 1 ) / actBatchSize );
        end
        
    end

end