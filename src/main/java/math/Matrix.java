package math;

public class Matrix {
    private double[] matrix;
    private int nRows;
    private int nCols;

    public Matrix(int n_rows, int n_cols) {
        nRows = n_rows;
        nCols = n_cols;

        matrix = new double[nRows * nCols];
    }

    public int getRows() {
        return nRows;
    }
    public int getCols() {
        return nCols;
    }

    public double[] getMatrix() {
        return matrix;
    }

    public double get(int idx1, int idx2) {
        return matrix[idx1 * nCols + idx2];
    }

    public void set(int idx1, int idx2, double val) {
        matrix[idx1 * nCols + idx2] = val;
    }

    public Matrix transpose() {
        Matrix m = new Matrix(nCols, nRows);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                m.set(j, i, this.get(i, j));
            }
        }
        return m;
    }

    public static Matrix add(Matrix a, Matrix b) {
        assert(a.getRows() == b.getRows() && a.getCols() == b.getCols());
        Matrix c = new Matrix(a.getRows(), a.getCols());
        for (int i = 0; i < a.getRows(); i++) {
            for (int j = 0; j < a.getCols(); j++) {
                c.set(i, j, a.get(i, j) + b.get(i, j));
            }
        }
        return c;
    }

    public static Matrix matrix_mult(Matrix a, Matrix b) {
        // If a is an n x m matrix and b is an m x l matrix, then ab is an n x l matrix
        assert(a.getCols() == b.getRows());
        Matrix c = new Matrix(a.getRows(), b.getCols());
        for (int i = 0; i < a.getRows(); i++) {
            for (int j = 0; j < b.getCols(); j++) {
                for (int k = 0; k < a.getCols(); k++) {
                    // c[i, j] += a[i, k] * b[k, j]
                    c.set(i, j, c.get(i, j) + a.get(i, k) * b.get(k, j));
                }
            }
        }
        return c;
    }

    public static Vector matrix_mult(Matrix a, Vector b) {
        // If a is an n x m matrix, b must be a vector of size m
        assert(a.getCols() == b.getSize());
        Vector c = new Vector(a.getRows());
        for (int i = 0; i < a.getRows(); i++) {
            for (int j = 0; j < a.getCols(); j++) {
                c.set(i, c.get(i) + a.get(i, j) * b.get(j));
            }
        }
        return c;
    }

    public static Matrix comp_matrix_mult(Matrix a, Matrix b) {
        assert(a.getRows() == b.getRows() && a.getCols() == b.getCols());
        Matrix c = new Matrix(a.getRows(), a.getCols());
        for (int i = 0; i < a.getRows(); i++) {
            for (int j = 0; j < b.getCols(); j++) {
                c.set(i, j, a.get(i, j) * b.get(i, j));
            }
        }
        return c;
    }
}
