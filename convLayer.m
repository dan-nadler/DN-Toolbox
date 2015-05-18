function layer = convLayer( kernelSize, featureNum, kernelFxn, poolingSize, poolingFxn )
% layer = convLayer( kernelSize, featureNum, kernelFxn, poolingSize, poolingFxn )

    layer.kSize = kernelSize;
    layer.nFeature = featureNum;
    layer.kFxn = kernelFxn;
    layer.pSize = poolingSize;
    layer.pFxn = poolingFxn;

end