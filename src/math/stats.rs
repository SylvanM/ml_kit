//!
//! A collection of statistics routines, like computing planes of best fits,
//! variances, and means
//! 

use matrix_kit::dynamic::matrix::Matrix;

use super::svd::compressed_svd;

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

/// With a $k$-dimensional plane of best fit, this predicts the missing 
/// component of a $k-1$ dimensional vector which comes from the 
/// distribution described by the plane of best fit.
pub fn predict(plane: (Matrix<f64>, Matrix<f64>), witness: Matrix<f64>) {
    
}