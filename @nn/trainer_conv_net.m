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
            
            % c_input{ic+1,1} is the output of layer ic, and is the input to the next layer
            [ c_input{ic+1,1}, Ac{ic} ] = obj.convolve( c_input{ic}, obj.Wc{ic}, ...
                obj.Bc{ic}, obj.convLayers{ic}.pSize );
            
        end
        
        input{1,1} = reshape( c_input{end}, actBatchSize, size(c_input{end},2) * size(c_input{end},3) ...
            * size(c_input{end},4) );
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
            % error = prior layer error * weight matrix .* dF/dA 
            errorOut{i-1,1} = errorOut{i} * obj.W{i}' .* dA{i} .* drop{i};
            hessian{i-1,1} = ( ...
                                    obj.F{i-1,2}( input{i} + obj.options.hessianStep * errorOut{i-1,1} ) ...
                                  - obj.F{i-1,2}( input{i} ) ...
                                )...
                                / obj.options.hessianStep;
        end
        
        % conv net backprop
        
        for icLayer = numConvLayers:-1:1
            
            if icLayer == numConvLayers
                c_error{icLayer,1} = errorOut{1} * obj.W{1}';
            else
                c_error{icLayer,1} = c_error{icLayer+1,1};
            end
            
            err = c_error{icLayer};
            weights = obj.Wc{icLayer};
            kSize = obj.convLayers{icLayer}.kSize;
            pSize = obj.convLayers{icLayer}.pSize;
            fNum = obj.convLayers{icLayer}.nFeature;

            obs = size( c_input{icLayer}, 1 );
            pts = size( c_input{icLayer}, 2 );
            chs = size( c_input{icLayer}, 3 );

            pOut = reshape( err, size(c_input{end}) );
            kron_vec = ones(1, (pts-kSize+1)/pSize ) / ((pts-kSize+1)/ pSize );

            c_errorOut{icLayer} = zeros( size( obj.Wc{icLayer} ) );

            sigm =  @(x) 1 ./ (1 + exp(-x) ); %sigmoid

            for ic = 1:chs

                for io = 1:obs

                    cIn = c_input{icLayer+1}(io,:,ic);
                    cA = sigm( cIn );
                    cdA = cA .* ( 1 - cA );

                    for fi = 1:fNum

                        cW = obj.Wc{icLayer}( fi, : );
                        upsampled_error = kron( pOut( io, :, fi, ic ), kron_vec );
%                         upsampled_input = kron( cIn, kron_vec );
                        
                        try
                            % mex file
                            [ temp, c_hessian{icLayer} ] = cnn_backprop( upsampled_error, Ac{icLayer}( io, :, fi, ic ), cIn ,cW );
                            c_errorOut{icLayer}( fi, : )  = c_errorOut{icLayer}( fi, : ) + temp;
                            clear temp;
                        catch
                            % Slow-ass Matlab loop :(
                            for iu = 1:numel( upsampled_error )

                                c_err =  upsampled_error(iu);
                                c_A = Ac{icLayer}( io, iu, fi, ic );
                                c_dA = c_A.*(1-c_A);

                                A_mult = c_input{icLayer}(io,1+iu-1:kSize+iu-1,ic);

                                c_errorOut{icLayer}( fi, : ) = ...
                                    c_errorOut{icLayer}( fi, : ) + (c_err * cW .* c_dA); % .* A_mult);

                            end
                        end

                    end

                end

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
        
        % update conv weights and biases
        
        for i = 1:numConvLayers
%             weight = old weight - learnRate * ( ( lambda * regularizer ) + ( activation * error ) ) 
            obj.Wc{i} = obj.Wc{i} - obj.options.learningRate * c_errorOut{i} / actBatchSize;
            
            obj.Bc{i} = obj.Bc{i} - obj.options.learningRate * ( sum( c_errorOut{i}, 2 ) / actBatchSize );
        end
        
        if obj.options.log
            obj.logs.rmse_batch(end+1,1) = obj.calcRMSE( A{numLayers+1}, obj.Y(s:e,:) );
            obj.logs.Wc{end+1,1} = obj.Wc{i};
            obj.logs.Bc{end+1,1} = obj.Bc{i};
        end
    
        
    end

end