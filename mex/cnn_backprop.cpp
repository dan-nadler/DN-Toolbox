/*==========================================================
* cnn_backprop.cpp -- performs backprop on a convolutional
*   /pooling layer pair
*
* This is a MEX-file for MATLAB. WIP.
*
* Currently performs the most inner loop of backprop for 
*  a sigmoid activation layer.
*
*========================================================*/

#include "mex.h"
#include <math.h>

/* kron */
void kron(double * kron_vector, double * source_vector, mwSize nK, mwSize nS, double * output) {
	mwSize i, j, k, N;

	N = nK * nS;
	k = 0;

	/* for each source vector cell */
	for (i = 0; i < nS; i++) {

		/* for each kron vector cell */
		for (j = 0; j < nK; j++) {

			/* output = kron cell x source cell */
			output[k] = source_vector[i] * kron_vector[j];
			k++;
		}

	}
}

/* upsamples the error and runs through the upsampled_error_loop function */
void upsample_pooling_loop( int pts, int kernel_size, int pool_size ) {

}

/* sigmoid activation function */
double sigmoid(double in) {
	double out;
	out = 1 / (1 + exp(-1 * in));
	return out;
}

/* Calculates kernel error, given activations and upsampled error */
void upsampled_error_loop( 
	double * upsampled_error, 
	double * activations, 
	double * weights, 
	double * inputs,
	mwSize upsamp_size, 
	mwSize kernel_size, 
	double * error_out, 
	double * hessian_out ) 
{
	mwSize i, j;
	double error_scalar;
	double step_size;

	step_size = 0.000001;

	/* for each upsampled point */
	for (i = 0; i < upsamp_size; i++) {
		error_scalar = upsampled_error[i] * (activations[i] * (1 - activations[i]));

		/* dot product of error scalar with kernel weights */
		for (j = 0; j < kernel_size; j++) {
			error_out[j] += error_scalar * weights[j];
		}
	}

	for (j = 0; j < kernel_size; j++) {
		hessian_out[j] = (sigmoid(inputs[j] + (step_size*error_out[j])) - sigmoid(inputs[j])) / step_size;
	}
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],	int nrhs, const mxArray *prhs[])
{
	double * upsampled_error;	/* 1xN input matrix */
	double * activations;		/* 1xN input matrix */
	double * weights;			/* 1xK input matrix */
	double * inputs;			/* 1xN input matrix */
	size_t ncols, kcols;		/* size of matrices */
	double * error_out;			/* output matrix	*/
	double * hessian_out;		/* output matrix	*/

	/* check for proper number of arguments */
	/*if (nrhs != 3) {
		mexErrMsgIdAndTxt("DNToolbox:cnnBackprop:nrhs", "Three inputs required.");
	}*/
	if (nlhs != 2) {
		mexErrMsgIdAndTxt("DNToolbox:cnnBackprop:nlhs", "Two output required.");
	}

	/* make sure the first input argument is type double */
	/*if (!mxIsDouble(prhs[0]) ||
		mxIsComplex(prhs[0])) {
		mexErrMsgIdAndTxt("DNToolbox:cnnBackprop:notDouble", "Input matrix must be type double.");
	}*/

	/* create a pointer to the real data in the input matrix  */
	upsampled_error = mxGetPr(prhs[0]);
	activations = mxGetPr(prhs[1]);
	inputs = mxGetPr(prhs[2]);
	weights = mxGetPr(prhs[3]);

	/* get dimensions of the input matrix */
	ncols = mxGetN(prhs[0]);
	kcols = mxGetN(prhs[3]);

	/* create the output matrix */
	plhs[0] = mxCreateDoubleMatrix(1, kcols, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, kcols, mxREAL);

	/* get a pointer to the real data in the output matrix */
	error_out = mxGetPr(plhs[0]);
	hessian_out = mxGetPr(plhs[1]);

	/* call the computational routine */
	upsampled_error_loop(upsampled_error, activations, weights, inputs, ncols, kcols, error_out, hessian_out);

}