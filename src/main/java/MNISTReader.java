import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class MNISTReader {
    public static void main(String[] args) {
        Matrix mat = new DenseMatrix(100, 100);
        System.out.printf("Matrix is here: rows(%d), cols(%d)\n", mat.numRows(), mat.numColumns());
    }
}