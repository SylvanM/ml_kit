//!
//! Mathematical Algorithms implementing Singular Value Decomposition
//! 
//! This uses the algorithms described in Golub and van Loan
//!

use matrix_kit::dynamic::matrix::Matrix;

/// Algorithm 5.1.3 from Golub and van Loan
/// 
/// See Golub and van Loan for a description. This helps compute Givens 
/// rotations designed to zero-out a certain coordinate.
/// 
/// This returns (c, s) where c = cos theta and s = sin theta.
fn determine_givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else {
        if b.abs() > a.abs() {
            let t = -a/b;
            let s = 1.0 / (1.0 + t.powf(2.0));
            let c = s * t;
            (c, s)
        } else {
            let t = -b/a;
            let c = 1.0 / (1.0 + t.powf(2.0));
            let s = c * t;
            (c, s)
        }
    }
}

/// Right-multiplies a matrix B with a givens rotation in the form of 
/// what appears in the Golub-Kahan step.
fn givens_rightmult(matrix: &mut Matrix<f64>, k: usize, c: f64, s: f64) {
    for i in 0..matrix.row_count() {
        let t1 = matrix.get(i, k);
        let t2 = matrix.get(i, k + 1);
        matrix.set(i, k, c * t1 - s * t2);
        matrix.set(i, k + 1, s * t1 + c * t2);
    }
}

/// Applies a Givens Transpose rotation to a matrix B, as in the form of 
/// the Golub-Kahan step.
/// 
/// This is NOT implemented efficiently.
fn givens_leftmult(matrix: &mut Matrix<f64>, k: usize, c: f64, s: f64) {
    for i in 0..matrix.row_count() {
        let t1 = matrix.get(k, i);
        let t2 = matrix.get(k + 1, i);
        matrix.set(k, i, c * t1 - s * t2);
        matrix.set(k + 1, i, s * t1 + c * t2);
    }
}

/// Algorithm 8.6.1 from Golub and van Loan
/// 
/// This takes in a bidiagonal matrix B, represented as a vector of the 
/// main diagonal, and another vector being the superdiagonal. The algorithm
/// outputs a bidiagonal matrix in the form of its diagonal and superdiagonal.
fn golub_kahan_step(diagonal: Vec<f64>, superdiagonal: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = diagonal.len();
    let m = n - 1;
    debug_assert_eq!(m, superdiagonal.len());

    // (Step 1) Solve the eigenvalues of the 2x2 trailing submatrix of T = B^T B

    let b = -diagonal[m - 1].powf(2.0) - superdiagonal[m - 2].powf(2.0)
            -diagonal[n - 1].powf(2.0) - superdiagonal[m - 1].powf(2.0);
    
    let c = (diagonal[m - 1].powf(2.0) * diagonal[n - 1].powf(2.0)) +
            (diagonal[n - 1].powf(2.0) * superdiagonal[m - 2].powf(2.0)) +
            (superdiagonal[m - 2].powf(2.0) * superdiagonal[m - 1].powf(2.0));

    let lambda_1 = (-b + (b.powf(2.0) - 4.0 * c).sqrt()) / 2.0;
    let lambda_2 = (-b - (b.powf(2.0) - 4.0 * c).sqrt()) / 2.0;

    let target = diagonal[n - 1].powf(2.0) + superdiagonal[m - 1].powf(2.0);
    let lambda = if (target - lambda_1).abs() < (target - lambda_2).abs() {
        lambda_1 
    } else { 
        lambda_2 
    };

    let mut y = diagonal[0].powf(2.0) - lambda;
    let mut z = diagonal[0] * superdiagonal[0];

    let mut b = Matrix::from_bidiagonal(diagonal, superdiagonal);

    for k in 0..(n - 1) {
        let (mut c, mut s) = determine_givens(y, z);
        givens_rightmult(&mut b, k, c, s);
        y = b.get(k, k);
        z = b.get(k + 1, k);
        (c, s) = determine_givens(y, z);
        givens_leftmult(&mut b, k, c, s);

        if k < n - 2 {
            y = b.get(k, k + 1);
            z = b.get(k, k + 2);
        }
    }

    (b.get_diagonal(), b.get_upperdiagonal())

}

#[cfg(test)]
mod svd_math_tests {
    use super::golub_kahan_step;


    #[test]
    fn test_golub_kahan_step() {
        let diag = vec![1.0, 3.0, 0.5, -4.0];
        let updi = vec![2.0, -0.3, -0.1];

        let b = golub_kahan_step(diag, updi);
        println!("{:?}", b);
    }

}