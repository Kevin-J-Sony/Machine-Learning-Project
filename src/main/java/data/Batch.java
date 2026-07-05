package data;

import math.Vector;

public class Batch {
    private Vector[] data;
    private int batchSize;

    public Batch(Vector[] input) {
        batchSize = input.length;

        data = input;
    }

    public Vector get(int index) {
        return data[index];
    }

    public Vector[] getData() {
        return data;
    }

    public int getSize() {
        return batchSize;
    }
}
