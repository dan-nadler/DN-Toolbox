function obj = trainer_backprop_quasi_newton( obj )   
    
    numLayers = size(obj.F,1);
    dloss = @(yh, y) (yh-y);
    reg = obj.regs.active;
    batchSize = obj.options.batchSize;
    
    b = 0;
    while batchSize*b < size(obj.X,1)
        
        b = b + 1;
        e = min( batchSize * b, size(obj.X,1) );
        s = batchSize * b - batchSize + 1;
        actBatchSize = e-s+1;

        % forward prop
        % input layer activation
        A{1,1} = obj.X(s:e,:); % input layer
        input{1,1} = A{1,1};
        
        for i = 1:numLayers
            % hidden and output layer activation
            input{i+1,1} = A{i} * obj.W{i} + repmat(obj.B{i},actBatchSize,1); %input to layer
            A{i+1,1} = obj.F{i,1}( input{i+1,1} ); % activation of layer
            dA{i+1,1} = obj.F{i,2}( input{i+1,1} ); % activation gradient of layer

        end

        % backprop
        % output layer error
        errorOut{numLayers,1} = dloss( A{numLayers+1}, obj.Y(s:e,:) );
        hessian{numLayers,1} = ( ...
                                    obj.F{numLayers,2}( input{numLayers+1} + obj.options.hessianStep * errorOut{numLayers,1} ) ...
                                  - obj.F{numLayers,2}( input{numLayers+1} ) ...
                                )...
                                / obj.options.hessianStep;
        
        for i = numLayers:-1:2
            % hidden layer error
            % error = prior layer error * weight matrix .* dF/dA .* current layer dF/dA
            errorOut{i-1,1} = errorOut{i} * obj.W{i}' .* dA{i} .* obj.F{i-1,2}( input{i} );
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
        
        obj.logs.rmse(end+1,1) = mean(( sum( errorOut{numLayers,1}, 1 ) ./ actBatchSize ).^2);
        
    end
    obj.logs.W{end+1,1} = obj.W;
    obj.logs.hessian{end+1,1} = hessian;

    if obj.options.visual == true
        plot([obj.Y(1,:)',obj.predict(obj.X(1,:))']);
        learnMovie(i) = getframe;
        pause(0.0001);
    end
    
end

function output = RvX( w, RvY, v, y )
    output = sum( ( w * RvY ) + ( v * y ) );
end

function output = RvY( fxn, x, RvX )
    output = fxn(x) * RvX;
end
