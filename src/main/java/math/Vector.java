package math;

public class Vector {
    private double[] array;
    private int size;

    public Vector(int s) {
        size = s;

        array = new double[size];
    }

    public int getSize() {
        return size;
    }

    public double[] getArray() {
        return array;
    }

    public double get(int idx) {
        return array[idx];
    }

    public void set(int idx, double value) {
        array[idx] = value;
    }

    public static Vector add(Vector a, Vector b) {
        assert(a.getSize() == b.getSize());
        Vector c = new Vector(a.getSize());
        for (int i = 0; i < a.getSize(); i++) {
            c.getArray()[i] = a.getArray()[i] + b.getArray()[i];
        }
        return c;
    }
}
