package ann;

import math.Matrix;
import math.Vector;

public class NeuralNetwork {
    private Layer[] layers;
    private int numbOfLayers;
    private double gamma = 0.5;

    public NeuralNetwork(int... layerDimensions) {
        numbOfLayers = layerDimensions.length;
        layers = new Layer[numbOfLayers - 1];

        for (int i = 0; i < numbOfLayers - 1; i++) {
            layers[i] = new Layer(layerDimensions[i], layerDimensions[i + 1]);
        }
    }

    public void train(Vector[] trainingDataInput, Vector[] trainingDataOutput) {
        assert(trainingDataInput.length == trainingDataOutput.length);
        int loops = 0;
        while (loops < 1000) {
            // for each input and output, pass through and adjust
            for (int idxOfInputs = 0; idxOfInputs < trainingDataInput.length; idxOfInputs++) {
                // store a list of intermediate outputs of each layer
                Vector[] linearIntermediateOutputs = new Vector[numbOfLayers];
                Vector[] yIntermediateOutputs = new Vector[numbOfLayers];
                Vector[] zIntermediateOutputs = new Vector[numbOfLayers];
                Vector currInputOfLayer = trainingDataInput[idxOfInputs];

                // Given that x is the input, W is the weight, b is the bias, and f is the activation function
                // we have: z[i] = W[i]x[i] + b[i], and y[i] = f(z[i])
                for (int i = 0; i < numbOfLayers; i++) {
                    linearIntermediateOutputs[i] = Matrix.matrix_mult(layers[i].getWeight(), currInputOfLayer);
                    zIntermediateOutputs[i] = Vector.add(linearIntermediateOutputs[i], layers[i].getBias());
                    yIntermediateOutputs[i] = layers[i].getActivationFunction().apply(zIntermediateOutputs[i]);
                    currInputOfLayer = yIntermediateOutputs[i];
                }

                // Now update the weights and biases, i.e. W[i] = W[i] + gamma * dL/dW[i] and b[i] = b[i] + gamma * dL/db[i]
                for (int i = numbOfLayers - 1; i >= 0; i++) {
                    // compute dL/dW[i]
                }
            }

            loops++;
        }
    }
}
