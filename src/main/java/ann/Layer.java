package ann;

import math.Matrix;
import math.Vector;

import java.util.function.Function;

public class Layer {
    private Matrix weight;
    private Vector bias;

    private int inputDim;
    private int outputDim;

    private Function<Vector, Vector> actFunc;
    private Function<Vector, Vector> actFuncDeriv;

    private static Vector sigmoid(Vector s) {
        Vector t = new Vector(s.getSize());
        for (int i = 0; i < s.getSize(); i++) {
            t.set(i, 1.0 / (1 + Math.exp(-s.get(i))) );
        }
        return t;
    }

    private static Vector sigmoidDeriv(Vector s) {
        Vector t = sigmoid(s);
        for (int i = 0; i < t.getSize(); i++) {
            t.set(i, t.get(i) * (1 - t.get(i)));
        }
        return t;
    }

    public Layer(int inputSize, int outputSize) {
        inputDim = inputSize;
        outputDim = outputSize;

        Matrix weight = new Matrix(outputDim, inputDim);
        Vector bias = new Vector(outputDim);

        double bd = Math.sqrt(6.0/(outputDim + inputDim));
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < inputDim; j++) {
                weight.set(i, j, Math.random() * 2 * bd - bd);
            }
        }

        actFunc = Layer::sigmoid;
        actFuncDeriv = Layer::sigmoidDeriv;
    }

    public Matrix getWeight() {
        return weight;
    }

    public Vector getBias() {
        return bias;
    }

    public Function<Vector, Vector> getActivationFunction() {
        return actFunc;
    }

    public Function<Vector, Vector> getActivationFunctionDerivative() {
        return actFuncDeriv;
    }

}
