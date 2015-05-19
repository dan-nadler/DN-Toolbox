function [ input, dW ] = provolve( err, weights, kSize, pSize, pts, dA )
    
    chs = size( err, 2 ) / pSize;
    obs = size( err, 1 );
    
    pOut = reshape( err, [obs, pSize, chs] );
    kron_vec = ones(1, (pts-kSize+1)/pSize ) / ((pts-kSize+1)/ pSize );
    
    upsample = nan( size( dA ) );
    numU = size( dA,2 );
    
    dW = zeros( size( weights ) );
    
    for ic = 1:chs
        
        W = weights( ic, : );
        
        for io = 1:obs
        
            upsample( io, :, ic ) = kron( pOut(io,:,ic), kron_vec ) .* dA( io, :, ic );
            
            for iu = 1:numU
                
                dW( ic, : ) = dW + 
                
            end
        
        end
        
    end
    input = 1;
end