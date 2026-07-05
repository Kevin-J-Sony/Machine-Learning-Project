import no.uib.cipr.matrix.DenseMatrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest {

    @Test
    public void testMatrixInitialization() {
        DenseMatrix matrix = new DenseMatrix(2, 2);

        // Set values: set(row, column, value)
        matrix.set(0, 0, 5.0);
        matrix.set(0, 1, 10.0);

        // Assertions verify the matrix behaves as expected
        assertEquals(2, matrix.numRows(), "Matrix should have 2 rows");
        assertEquals(2, matrix.numColumns(), "Matrix should have 2 columns");
        assertEquals(5.0, matrix.get(0, 0), 0.001, "Value at (0,0) should be 5.0");
        assertEquals(10.0, matrix.get(0, 1), 0.001, "Value at (0,1) should be 10.0");
        assertEquals(0.0, matrix.get(1, 0), 0.001, "Unset values should default to 0.0");
        assertEquals(0.0, matrix.get(1, 1), 0.001, "Unset values should default to 0.0");
    }
}
