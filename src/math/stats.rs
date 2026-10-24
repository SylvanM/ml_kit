//!
//! A collection of statistics routines, like computing planes of best fits,
//! linear regressions, variances, and means
//! 

use matrix_kit::dynamic::matrix::Matrix;

use super::svd::{compressed_svd, house, svd};

/// Computes the mean of a set of data points, encoded as the columns of a 
/// matrix. A vector representing the midpoint is returned.
pub fn mean(data: Matrix<f64>) -> Matrix<f64> {
    let ones = Matrix::ones(data.col_count(), 1);
    let mut mean = data.clone() * ones;
    mean /= data.col_count() as f64;
    mean
}

/// Takes a set of points and centers them at the origin by subtracting 
/// their mean, returning their centered versions and the mean that was 
/// subtracted.
pub fn centered(data: Matrix<f64>) -> (Matrix<f64>, Matrix<f64>) {
    let mean = mean(data.clone());
    let subtraction_matrix = Matrix::from_cols(vec![mean.clone() ; data.col_count()]);
    (data - subtraction_matrix, mean)
}

/// Returns the $k$-dimensional plane of best fit. When $k$ is 1, this is 
/// the line of best fit. This is returned in the form of a matrix whose 
/// columns for the basis of the $k$-plane of best fit, and a vector
/// which is the "offset" from the origin. The basis is an orthonormal basis.
pub fn plane_of_best_fit(
    data: Matrix<f64>, k: usize
) -> (Matrix<f64>, Matrix<f64>) {
    let (centered, mean) = centered(data);
    let (_, v, _) = compressed_svd(&centered.transpose(), k);
    let basis = v;
    (basis, mean)
}

/// Computes the linear regression (ordinary least squares) of `y` on `x`. As
/// with the other functions here, the data points are the columns of `x`, and
/// `y` is a row vector holding the value for each data point.
///
/// Returns `(w, b)`, where `w` is a row vector, so that `w * x + b` is as close
/// as possible to `y`. If several `w` fit equally well (for example, when one
/// row of `x` is a multiple of another), this returns the shortest one.
pub fn linear_regression(x: Matrix<f64>, y: Matrix<f64>) -> (Matrix<f64>, f64) {
    let (k, n) = (x.row_count(), x.col_count());
    assert!(k > 0 && n > 0, "linear_regression needs at least one data point and one feature");
    assert_eq!((y.row_count(), y.col_count()), (1, n), "y needs exactly one value per column of x");

    // Centering the data takes care of the intercept, b
    let (x_centered, x_mean) = centered(x);
    let (y_centered, y_mean) = centered(y);

    // We want the w minimizing |A w^T - c|, where A = x_centered^T has a row per
    // data point. Running the SVD on A directly would be slow when there are many
    // data points, so first shrink it to the k x k matrix R from its QR
    // factorization, using Householder reflections. Reflecting c along with A
    // doesn't change the answer.
    let mut a = x_centered.transpose();
    let mut c = y_centered.transpose();
    for j in 0..k.min(n) {
        let (v, beta) = house(a.get_submatrix(j..n, j..(j + 1)));

        let a_block = a.get_submatrix(j..n, j..k);
        a.set_submatrix(j..n, j..k, a_block.clone() - v.clone() * (v.transpose() * a_block) * beta);

        let c_block = c.get_submatrix(j..n, 0..1);
        c.set_submatrix(j..n, 0..1, c_block.clone() - v.clone() * (v.transpose() * c_block) * beta);
    }
    let r = a.get_submatrix(0..k.min(n), 0..k);
    let c = c.get_submatrix(0..k.min(n), 0..1);

    // With R = U S V^T, the least squares solution is w^T = V S^+ U^T c, where S^+
    // inverts the singular values, except ones so small they're just rounding
    // error, which it sets to 0
    let (u, v, s) = svd(&r);
    assert!(u.row_count() > 0, "SVD did not converge");
    let singular_values = s.get_diagonal();
    let tolerance = (n.max(k) as f64) * f64::EPSILON * singular_values[0];

    let u_t_c = u.transpose() * c;
    let s_inv_u_t_c = Matrix::from_index_def(k, 1, &mut |i, _| {
        if i < singular_values.len() && singular_values[i] > tolerance {
            u_t_c.get(i, 0) / singular_values[i]
        } else {
            0.0
        }
    });
    let w = (v * s_inv_u_t_c).transpose();

    let b = y_mean.get(0, 0) - (w.clone() * x_mean).get(0, 0);
    (w, b)
}

#[cfg(test)]
mod stats_tests {
    use matrix_kit::dynamic::matrix::Matrix;

    use super::{linear_regression, plane_of_best_fit};


    #[test]
    fn test_simple_line() {

        let data = Matrix::from_flatmap(2, 7, vec![
            7.0, 15.0,
            3.0, 12.0,
            9.0, 12.0,
            10.0, 13.0,
            1.0, 11.0,
            9.0, 14.0,
            5.0, 13.0,
        ]);

        let (line, mean) = plane_of_best_fit(data, 1);

        println!("Line spanned by {:?}\nwith mean {:?}", line, mean);

        println!("Slope is {:?}", line.get(1, 0) / line.get(0, 0));

    }

    #[test]
    fn test_linear_regression() {
        // These points lie exactly on the plane y = 3 + 2a - 5b
        let x = Matrix::from_flatmap(2, 6, vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            2.0, 3.0,
            -1.0, 4.0,
            5.0, -2.0,
        ]);
        let y = Matrix::from_flatmap(1, 6, vec![3.0, 5.0, -2.0, -8.0, -19.0, 23.0]);

        let (w, b) = linear_regression(x, y);

        assert!((w.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((w.get(0, 1) + 5.0).abs() < 1e-10);
        assert!((b - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_regression_line() {
        // The points from test_simple_line, regressing the second coordinate on
        // the first. plane_of_best_fit minimizes the perpendicular distances to
        // the points, but a regression minimizes the vertical ones, so this should
        // match the textbook formulas: the slope is S_xy / S_xx = 107 / 486, and
        // the line goes through the mean point, (44/7, 90/7).
        let x = Matrix::from_flatmap(1, 7, vec![7.0, 3.0, 9.0, 10.0, 1.0, 9.0, 5.0]);
        let y = Matrix::from_flatmap(1, 7, vec![15.0, 12.0, 12.0, 13.0, 11.0, 14.0, 13.0]);

        let (w, b) = linear_regression(x, y);

        let slope = 107.0 / 486.0;
        assert!((w.get(0, 0) - slope).abs() < 1e-10);
        assert!((b - (90.0 / 7.0 - slope * 44.0 / 7.0)).abs() < 1e-10);
    }

    #[test]
    fn test_regression_repeated_feature() {
        // The second feature is always twice the first, so every w with
        // w_1 + 2 w_2 = 1 fits y exactly. The shortest of them is (0.2, 0.4).
        let x = Matrix::from_flatmap(2, 4, vec![
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
            5.0, 10.0,
        ]);
        let y = Matrix::from_flatmap(1, 4, vec![1.0, 2.0, 3.0, 5.0]);

        let (w, b) = linear_regression(x, y);

        assert!((w.get(0, 0) - 0.2).abs() < 1e-10);
        assert!((w.get(0, 1) - 0.4).abs() < 1e-10);
        assert!(b.abs() < 1e-10);
    }

}