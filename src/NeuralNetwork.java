class NeuralNetwork {
    int input, hidden, output;


    matrix weights_IH, weights_HO, bias_H, bias_O;

    NeuralNetwork ( int in_, int hidden_, int out_) {
        input = in_;
        hidden = hidden_;
        output = out_;

        weights_IH = new matrix(hidden, input);    // weights between in and hidden
        weights_HO = new matrix(output, hidden);   // weights between hidden and out

        weights_IH.randomise();
        weights_HO.randomise();

        bias_H = new matrix(hidden, 1);  //weights for the bias for the hidden;
        bias_O = new matrix(output, 1);  //weights for the bias for the output;

        bias_H.randomise();
        bias_O.randomise();
    }

    float[] predict(float[] inArray_) {
        // generate hidden outputs
        matrix inMatrix = new matrix(inArray_.length, 1);
        inMatrix.fromArray(inArray_);   //   this is now our inpus matrix
        // all inputs multiplied by weights
        matrix hidden = new matrix(matrix.multiply(weights_IH, inMatrix));
        // add in the bias weights
        hidden.addMatrix(bias_H);

        // activation function on all weights (squish them to 0-1)
        hidden.sigmoid();
        // Take these squished inputs and multiply them by the hidden wights
        matrix outMatrix = new matrix(matrix.multiply(weights_HO, hidden));
        // add in the output bias
        outMatrix.addMatrix(bias_O);
        //squish it again. This is now our guess. Between 0-1
        outMatrix.sigmoid();
        // return as an array
        float[] ffArray = new float[output];
        ffArray = matrix.toArray(outMatrix);
        // Just need to normalise probabilities to total 1
        float sum = 0;
        for (int i= 0;i <ffArray.length; i++) {
            ffArray[i] = (float) Math.exp( (float) ffArray[i]);
            sum = sum + ffArray[i];
        }
        for (int i= 0;i <ffArray.length; i++) {
            ffArray[i] = ffArray[i] / sum;

        }
        return ffArray;
    }
    void train (float[] inArray_, float[] targets_, float lr_) {
        // ************  From here ***********************************
        // *********** is really just the feed forward bit ***********
        // generate hidden outputs
        matrix inputs = new matrix(inArray_.length, 1);
        inputs.fromArray(inArray_);   //   this is now our inpus matrix
        // all inputs multiplied by weights
        matrix hidden = new matrix(matrix.multiply(weights_IH, inputs));
        // Add in the bias weight
        hidden.addMatrix(bias_H);

        // activation function squish to 0-1
        hidden.sigmoid();
        // Generating output outputs.
        matrix outMatrix = new matrix(matrix.multiply(weights_HO, hidden));
        // Take the squished hidden and multiply by the hidden weights.
        outMatrix.addMatrix(bias_O);
        //Squish it
        outMatrix.sigmoid();

        // ************  to here  **************************************
        // Now we need to do the back propogation
        // Get the targets as a matrix
        matrix matrixTargets  = new matrix(targets_.length, 1);
        matrixTargets.fromArray(targets_)  ;   // targets as a matrix
        // This gives us our output errors matrix.
        matrix output_errors = new matrix (matrix.subMatrix(matrixTargets, outMatrix));

        // Now we have our output errors we need the gradient descent bit.
        // sigmoid derivative
        outMatrix.dsigmoid();
        // multiplied by the output errors
        outMatrix.multiply(output_errors);
        // multiplied by the learning rate
        outMatrix.multiply(lr_);
        //update bias
        bias_O.addMatrix(outMatrix);

        //calculate delas
        matrix hidden_T = matrix.transpose(hidden);
        matrix weight_HO_deltas = new matrix(matrix.multiply(outMatrix, hidden_T));

        //Adjust delats
        weights_HO.addMatrix(weight_HO_deltas);

        // transpose Hidden to output Weights
        matrix weights_HO_T = new matrix(matrix.transpose(weights_HO));
        // get the hidden layers errors.
        matrix hidden_errors = new matrix(matrix.multiply(weights_HO_T, output_errors));
        // hidden gradient
        // sigmoid derivative
        hidden.dsigmoid();
        hidden.multiply(hidden_errors);
        hidden.multiply(lr_);
        // adjust bias
        bias_H.addMatrix(hidden);

        //  input to hidden
        matrix inputs_T = new matrix(matrix.transpose(inputs));
        matrix weight_IH_deltas = new matrix(matrix.multiply(hidden,inputs_T));

        // adjust
        weights_IH.addMatrix(weight_IH_deltas);


    }
}