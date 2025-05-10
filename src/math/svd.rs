//!
//! Mathematical Algorithms implementing Singular Value Decomposition
//! 
//! This uses the algorithms described in Golub and van Loan
//!

use core::f64;

use matrix_kit::dynamic::matrix::Matrix;
use rand_distr::num_traits::Pow;

/// Algorithm 5.1.1 (Householder vector) from Golub and van Loan
fn house(x: Matrix<f64>) -> (Matrix<f64>, f64) {
    let sub_x = Matrix::from_index_def(x.row_count() - 1, 1, 
        &mut |r, _| x.get(r + 1, 0));
    let sigma = sub_x.inner_product(&sub_x);
    let mut v = Matrix::identity(1, 1);
    v.append_mat_bottom(sub_x);

    let beta;
    let x0 = x.get(0, 0);

    if sigma == 0.0 && x0 >= 0.0 {
        beta = 0.0;
    } else if sigma == 0.0 && x0 < 0.0 {
        beta = -2.0;
    } else {
        let mu = (x0.powf(2.0) + sigma).sqrt();
        if x0 <= 0.0 {
            v.set(0, 0, x0 - mu);
        } else {
            v.set(0, 0, -sigma / (x0 + mu));
        }

        beta = 2.0 * v.get(0, 0).powf(2.0) / (sigma + v.get(0, 0).powf(2.0));

        v /= v.get(0, 0)
    }

    (v, beta)
}

/// Algorithm 5.4.2 (Householder Bidiagonalization)
/// 
/// Takes in an m x n matrix, with m >= n, returns matrices U, V, and B such that 
/// 
/// U^T * A * V = B
///               0
fn householder_bidiag(matrix: Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    let mut work_matrix = matrix.clone();

    let n = work_matrix.col_count();
    let m = work_matrix.row_count();

    let mut big_u = Matrix::identity(m, m);
    let mut big_v = Matrix::identity(n, n);

    for j in 0..n {
        let (mut v, mut beta) = house(work_matrix.get_submatrix(j..m, j..(j + 1)));
        work_matrix.set_submatrix(j..m, j..n, 
            (Matrix::identity(m - j, m - j) 
                - (v.clone() * v.transpose()) * beta)
                * work_matrix.get_submatrix(j..m, j..n)
        );

        // Instead of storing the essential part of the householder vector, 
        // we'll just continue computing U by accomulation of the intermediate
        // householder matrices
        let mut householder_vec = Matrix::new(m, 1);
        householder_vec.set(j, 0, 1.0);
        householder_vec.set_submatrix((j + 1)..m, 0..1, 
            v.get_submatrix(1..(m - j), 0..1)
        );
        big_u -= (big_u.clone() * householder_vec.clone()) * (householder_vec * beta).transpose();

        // zero-out the entries below the diagonal in the j-th column
        work_matrix.set_submatrix((j + 1)..m, j..(j + 1), 
            Matrix::new(m - j - 1, 1)
        );

        if j + 3 <= n {
            (v, beta) = house(work_matrix.get_submatrix(
                j..(j + 1), (j + 1)..n).transpose());
            work_matrix.set_submatrix(j..m, (j + 1)..n, 
                work_matrix.get_submatrix(j..m, (j + 1)..n) * 
                    (Matrix::identity(n - j - 1, n - j - 1) 
                        - (v.clone() * v.transpose() * beta))
            );

            householder_vec = Matrix::new(n, 1);
            householder_vec.set(j + 1, 0, 1.0);
            householder_vec.set_submatrix((j + 2)..n, 0..1, 
                v.get_submatrix(1..(n - j - 1), 0..1)
            );
            big_v -= (big_v.clone() * householder_vec.clone()) * (householder_vec * beta).transpose();

            // Zero out entries after the superdiagonal
            work_matrix.set_submatrix(j..(j + 1), (j + 2)..n, 
                Matrix::new(1, n - j - 2)
            );
        }
    }

    (big_u, big_v, work_matrix.get_submatrix(0..n, 0..n))
}

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
            let s = 1.0 / (1.0 + t.powf(2.0)).sqrt();
            let c = s * t;
            (c, s)
        } else {
            let t = -b/a;
            let c = 1.0 / (1.0 + t.powf(2.0)).sqrt();
            let s = c * t;
            (c, s)
        }
    }
}

/// Right-multiplies a matrix `B` with a givens rotation in the form of 
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
    for i in 0..matrix.col_count() {
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
/// 
/// This follows the implementation guide found here:
/// https://www.cs.utexas.edu/~inderjit/public_papers/HLA_SVD.pdf
/// 
/// This also updates the u and v matrices
fn golub_kahan_step(
    b: &mut Matrix<f64>, 
    u: &mut Matrix<f64>, v: &mut Matrix<f64>,
    q: usize, p: usize
) -> (Vec<f64>, Vec<f64>) {

    let n = b.col_count();

    // (Step 1) Solve the eigenvalues of the 2x2 trailing submatrix of T = B_22^T B_22
    let b22 = b.get_submatrix(p..(n - q), p..(n - q));
    let (diagonal, superdiagonal) = (b22.get_diagonal(), b22.get_upperdiagonal());

    let n_b22 = diagonal.len();
    let m_b22 = superdiagonal.len();

    let lambda = { // a 2x2 matrix, so we'll do something different.
        
        let b = if m_b22 == 1 {
            -diagonal[0].powf(2.0) - superdiagonal[0].powf(2.0) 
            -diagonal[1].powf(2.0)
        } else {
            -diagonal[m_b22 - 1].powf(2.0) - superdiagonal[m_b22 - 2].powf(2.0)
            -diagonal[n_b22 - 1].powf(2.0) - superdiagonal[m_b22 - 1].powf(2.0)
        };

        let c = if m_b22 == 1 {
            (diagonal[0].powf(2.0) * superdiagonal[0].powf(2.0)) +
            (diagonal[1].powf(2.0) * diagonal[0].powf(2.0)) - 
            (superdiagonal[0].powf(2.0) * diagonal[0] * superdiagonal[0])
        } else {
            (diagonal[m_b22 - 1].powf(2.0) * diagonal[n_b22 - 1].powf(2.0)) +
            (diagonal[n_b22 - 1].powf(2.0) * superdiagonal[m_b22 - 2].powf(2.0)) +
            (superdiagonal[m_b22 - 2].powf(2.0) * superdiagonal[m_b22 - 1].powf(2.0))
        };

        let lambda_1 = (-b + (b.powf(2.0) - 4.0 * c).sqrt()) / 2.0;
        let lambda_2 = (-b - (b.powf(2.0) - 4.0 * c).sqrt()) / 2.0;

        let target = diagonal[n_b22 - 1].powf(2.0) + superdiagonal[m_b22 - 1].powf(2.0);

        if (target - lambda_1).abs() < (target - lambda_2).abs() {
            lambda_1 
        } else { 
            lambda_2 
        }

    };

    let mut y = diagonal[0].powf(2.0) - lambda;
    let mut z = diagonal[0] * superdiagonal[0];

    for k in p..(n - q - 1) {
        let (mut c, mut s) = determine_givens(y, z);
        givens_rightmult(b, k, c, s);
        givens_rightmult(v, k, c, s);
        y = b.get(k, k);
        z = b.get(k + 1, k);
        (c, s) = determine_givens(y, z);
        givens_leftmult(b, k, c, s);
        givens_rightmult(u, k, c, s);

        if k < n - 2 {
            y = b.get(k, k + 1);
            z = b.get(k, k + 2);
        }
    }

    (b.get_diagonal(), b.get_upperdiagonal())

}

/// Computes the singular value decomposition of this matrix, `A`, returning 
/// a tuple `(U, V, S)` where `S` is a `Vec<f64>` of the singular values of
/// `A`, and
/// 
/// U^T * A * V = diag(S)
/// 
/// This is an implementation of Algorithm 8.6.2 from Golub and van Loan
pub fn svd_factorization(matrix: Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {

    let m = matrix.row_count();
    let n = matrix.col_count();

    debug_assert!(m >= n);

    let (mut u, mut v, mut b) = householder_bidiag(matrix.clone());

    let mut q = 0;

    while q < n {
        // For numerical stability, zero out some stuff
        for i in 0..(n - 1) {
            if b.get(i, i + 1).abs() <= f64::EPSILON * (b.get(i, i).abs() + b.get(i + 1, i + 1).abs()) {
                b.set(i, i + 1, 0.0);
            }
        }

        q = 0; // we may be able to range from q..(n - 1) instead
        for i in 0..(n - 1) {
            if b.get(n - i - 2, n - i - 1) != 0.0 {
                break;
            } else {
                q = i + 1;
            }
        }
        if q == n - 1 {
            q = n;
        }

        // now find smallest p!
        let mut p = n - q;
        for i in (1..(n - q)).rev() {
            if b.get(i - 1, i) != 0.0 {
                p = i - 1;
            } else {
                break;
            }
        }

        if q < n {
            // If any diagonal entry of B22 is zero, zero-out the superdiagonal
            // entry in the same row
            let mut found_zero_diag = false;
            for i in p..(n - q) {
                if b.get(i, i) == 0.0 { 
                    found_zero_diag = true;
                    if i < n - 1 {
                        b.set(i, i + 1, 0.0);
                    }
                }
            }

            if !found_zero_diag {
                golub_kahan_step(&mut b, &mut u, &mut v, q, p);
            }
        }
    }

    b.append_mat_bottom(Matrix::new(m - n, n));
    (u, v, b)
}

#[cfg(test)]
mod svd_math_tests {
    use matrix_kit::dynamic::matrix::Matrix;

    use crate::math::svd::householder_bidiag;

    use super::svd_factorization;


    #[test]
    fn test_bidiagonalization() {
        let a = Matrix::random_normal(5, 4, 0.0, 1.0);
        println!("{:?}", a);
        let (u, v, b) = householder_bidiag(a.clone());
        println!("{:?}\n{:?}\n{:?}", u, v, b);

        println!("{:?}", u.transpose() * a * v);
    }

    #[test]
    fn test_svd() {
        let a = Matrix::random_normal(5, 4, 0.0, 1.0);

        let (u, v, s) = svd_factorization(a.clone());

        // Are u and v orthogonal?

        println!("A: {:?}", a);
        println!("U S V^T: {:?}", u.clone() * s * v.transpose());

        // Yay it actually works!
    }
}