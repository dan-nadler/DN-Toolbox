function convolve2( filterDim, numFilters, input, W, b )

    numIn = size( input, 1 );
    inDim = size( input, 2 );
    convDim = inDim - filterDim + 1;
    
    sigm =  @(x) 1 ./ (1 + exp(-x) ); %sigmoid
    
    convolvedFeatures = zeros( convDim, numFilters, numIn );
    
    for inNum = 1:numIn
        
        in = in( inNum, : );
        
        for filterNum = 1:numFilters
           
            convolvedIn = zeros(1, convDim);
            
            filter = W( :, filterNum );
            filter = rot90(filter,2);
            
            convolvedIn = convolvedIn + conv2( in, filter, 'valid' );
            
            convolvedIn = sigm( convolvedIn + b( filterNum ) );
            
            
        end
        
    end

end