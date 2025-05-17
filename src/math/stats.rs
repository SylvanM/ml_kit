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

#[cfg(test)]
mod stats_tests {
    use matrix_kit::dynamic::matrix::Matrix;

    use super::plane_of_best_fit;


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

}