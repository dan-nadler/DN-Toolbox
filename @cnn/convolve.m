function [ output, A ] = convolve( input, W, B, pSize )
% convolve( input, weights, kSize, pSize )
% performs mean convolution -> sigmoid activation -> mean pooling

    sigm =  @(x) 1 ./ (1 + exp(-x) ); %sigmoid
    
    % assume input size: observations x time-series points x channels
    obs = size( input, 1 ); % this is essentially the batch size
    pts = size( input, 2 ); % the number of data-points for each channel
    chs = size( input, 3 ); % the number of channels 
    
    kSize = size( W, 2 );
    fNum = size( W, 1 );
    
    P = nan([obs,pSize,fNum,chs]);
    
    convDim = pts - kSize + 1;
    
    output = nan( obs, pSize, chs, fNum );
    
    convolved_features = zeros( [obs, convDim, fNum, chs] );
    
    for ic = 1:chs
        
        in_c = squeeze( input( :, :, ic ) );

        for io = 1:obs
            
            in_o = in_c( io, : );
            
            for fi = 1:fNum
                
                % get the kernel weights
                filter = rot90( W(fi,:), 2 );
                
                % convolution
                convolved_vec = sigm( ( conv2( in_o, filter, 'valid' ) / kSize ) + B(fi) );
                
                convolved_features( io, :, fi, ic ) = convolved_vec;
                
                
            end
             
            % pooling layer ouput, size  < obs x pSize x fNum x chs> 
            P( io, :, :, ic ) = mean( ...
                    reshape( convolved_features( io, :, :, ic ), convDim / pSize, pSize, fNum ) ...
                , 1);
            
        end

    end
    
    output = P;
    A = convolved_features;

end