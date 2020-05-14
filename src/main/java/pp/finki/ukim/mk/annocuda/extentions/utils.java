package pp.finki.ukim.mk.annocuda.extentions;

public class utils {

    public static double[] toRowMajor(double[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        double[] result = new double[m * n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(mat[i], 0, result, i * m, n);
        }
        return result;
    }

    public static double[][] toMatrix(double[] array, int m, int n){
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            if (n >= 0) System.arraycopy(array, i * m, result[i], 0, n);
        }
        return result;
    }
}
