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

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	double * kron_vector;		/* 1xN input matrix */
	double * source_vector;		/* 1xN input matrix */
	size_t nK, nS;				/* size of matrices */
	double * output;			/* output matrix */

	/* check for proper number of arguments */
	if (nrhs != 2) {
		mexErrMsgIdAndTxt("DNToolbox:cnnBackprop:nrhs", "Two inputs required.");
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
	kron_vector = mxGetPr(prhs[0]);
	source_vector = mxGetPr(prhs[1]);

	/* get dimensions of the input matrix */
	nK = mxGetN(prhs[0]);
	nS = mxGetN(prhs[1]);

	/* create the output matrix */
	plhs[0] = mxCreateDoubleMatrix(1, nS * nK, mxREAL);

	/* get a pointer to the real data in the output matrix */
	output = mxGetPr(plhs[0]);

	/* call the computational routine */
	kron(kron_vector, source_vector, nK, nS, output );
	

}