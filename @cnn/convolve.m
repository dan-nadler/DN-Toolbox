function [ output, dA, A ] = convolve( input, weights, kSize, pSize )
% convolve( input, weights, kSize, pSize )
% performs mean convolution -> sigmoid activation -> mean pooling

    sigm =  @(x) 1 ./ (1 + exp(-x) ); %sigmoid
    
    % assume input size: observations x time-series points x channels
    obs = size( input, 1 ); % this is essentially the batch size
    pts = size( input, 2 ); % the number of data-points for each channel
    chs = size( input, 3 ); % the number of channels
     
    C_index = conv2( ones( 1, size( input, 2 ) ), ones( 1, kSize ), 'same' ) == kSize;
    output = nan( obs, pSize, chs );
    
    for ic = 1:chs
        
        conv_vec = weights(ic,:); % weights is matrix of size ( maps x kSize )
        C_temp = nan( obs, pts-kSize+1 );

        for io = 1:obs
            
            % convolution
            C = conv2( input(io,:,ic), conv_vec, 'same' ); % vector 1 x pts
            C_temp(io,:) = C( :, C_index ); % vector 1 x pts - kSize + 1
            
        end
        
        % non-linearity and its gradient
        A = sigm( C_temp ); % vector 1 x pts - kSize + 1
        dA( :, :, ic ) = A .* (1 - A); % save this for backprop
        
        % mean pooling
        P = mean( reshape( A', size(A,2)/pSize, pSize, obs ) ); % vector 1 x pSize

        % output of pooling layer
        output( :, :, ic ) = permute(P(1,:,:),[3 2 1]); % final size is: obs x pSize

    end

end