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

/* Calculates kernel error, given activations and upsampled error */
void upsampled_error_loop( double * upsampled_error, double * activations, double * weights,
	mwSize upsamp_size, mwSize kernel_size, double * error_out) 
{
	mwSize i, j;
	double error_scalar;

	/* for each upsampled point */
	for (i = 0; i < upsamp_size; i++) {
		error_scalar = upsampled_error[i] * (activations[i] * (1 - activations[i]));

		/* dot product of error scalar with kernel weights */
		for (j = 0; j < kernel_size; j++) {
			error_out[j] += error_scalar * weights[j];
		}
	}
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	double multiplier;						/* input scalar */
	double * upsampled_error, * activations, * weights;	/* 1xN input matrices */
	size_t ncols, kcols;							/* size of matrices */
	double * error_out;						/* output matrix */

	/* check for proper number of arguments */
	if (nrhs != 3) {
		mexErrMsgIdAndTxt("DNToolbox:cnnBackprop:nrhs", "Three inputs required.");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("DNToolbox:cnnBackprop:nlhs", "One output required.");
	}

	/* make sure the first input argument is type double */
	if (!mxIsDouble(prhs[0]) ||
		mxIsComplex(prhs[0])) {
		mexErrMsgIdAndTxt("DNToolbox:cnnBackprop:notDouble", "Input matrix must be type double.");
	}

	/* create a pointer to the real data in the input matrix  */
	upsampled_error = mxGetPr(prhs[0]);
	activations = mxGetPr(prhs[1]);
	weights = mxGetPr(prhs[2]);

	/* get dimensions of the input matrix */
	ncols = mxGetN(prhs[0]);
	kcols = mxGetN(prhs[2]);

	/* create the output matrix */
	plhs[0] = mxCreateDoubleMatrix(1, kcols, mxREAL);

	/* get a pointer to the real data in the output matrix */
	error_out = mxGetPr(plhs[0]);

	/* call the computational routine */
	upsampled_error_loop(upsampled_error, activations, weights, ncols, kcols, error_out);

}