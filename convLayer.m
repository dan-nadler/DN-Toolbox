function layer = convLayer( kernelSize, featureNum, kernelFxn, activationFxn, poolingSize, poolingFxn )
%convLayer( kernelSize, featureNum, kernelFxn, activationFxn, poolingSize, poolingFxn )

    layer.kSize = kernelSize;
    layer.nFeature = featureNum;
    layer.kFxn = kernelFxn;
    layer.aFxn = activationFxn;
    layer.pSize = poolingSize;
    layer.pFxn = poolingFxn;

end