function obj = trainer_backprop_quasi_newton( obj )   
    
    numLayers = size(obj.F,1);
    loss = obj.loss;
    reg = obj.regs.active;
    batchSize = obj.options.batchSize;
    dropout = @(mat) rand(size(mat)) > obj.options.dropoutProb;
    
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
            % hidden and output layer activation
            input{i+1,1} = A{i} * obj.W{i} + repmat(obj.B{i},actBatchSize,1); %input to layer
            drop{i+1,1} = dropout( input{i+1} );
            A{i+1,1} = obj.F{i,1}( input{i+1,1} ) .* drop{i+1}; % activation of layer
            dA{i+1,1} = obj.F{i,2}( input{i+1,1} ) .* drop{i+1}; % activation gradient of layer

        end

        % backprop
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

        % update weights and biases
        for i = 1:numLayers
            % weight = old weight - learnRate * ( ( lambda * regularizer ) + ( activation * error ) ) 
            obj.W{i} = obj.W{i} - obj.options.learningRate * ...
                ( ( repmat( 0.01 * reg(obj.W{i}), 1, obj.N{i} ) .* sign(obj.W{i} ) ...
                + ( A{i}' * errorOut{i} ) + ( A{i}' * hessian{i} ) ) ) ...
                / actBatchSize;
            
            obj.B{i} = obj.B{i} - obj.options.learningRate * ( sum( errorOut{i}, 1 ) / actBatchSize );
        end

        if obj.options.log
            obj.logs.rmse_batch(end+1,1) = obj.calcRMSE( A{numLayers+1}, obj.Y(s:e,:) );
        end
        
    end
    
    if obj.options.log
        obj.logs.W{end+1,1} = obj.W;
        obj.logs.B{end+1,1} = obj.B;
        obj.logs.hessian{end+1,1} = hessian;
    end
    
    if obj.options.visual == true
        plot([obj.Y(1,:)',obj.propOutputFromInput(obj.X(1,:))']);
        learnMovie(i) = getframe;
        pause(0.0001);
    end
    
end
