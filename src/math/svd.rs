//!
//! Mathematical Algorithms implementing Singular Value Decomposition
//! 
//! This uses the algorithms described in Golub and van Loan
//!

use matrix_kit::dynamic::matrix::Matrix;

/// Algorithm 5.1.1 (Householder vector) from Golub and van Loan
fn house(x: Matrix<f64>) -> (Matrix<f64>, f64) {
    let sub_x = Matrix::from_index_def(x.row_count() - 1, 1, 
        &mut |r, _| x.get(r + 1, 0));
    let sigma = sub_x.inner_product(&sub_x);
    let mut v = Matrix::identity(1, 1);
    v.append_mat_bottom(sub_x);

    let mut beta = 0.0;
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
/// Takes in an m x n matrix, returns matrices U, V, and B such that 
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
        println!("{:?}", big_u);
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

        if j <= n - 3 {
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

    (big_u, big_v, work_matrix)

    // Now see if we can decompose it into the U and V matrices?

    // retrieve the essential part of the householder vector for U_j
    // let mut essential_u_vecs = vec![Matrix::new(m, 1) ; n];
    // let mut essential_v_vecs = vec![Matrix::new(n, 1) ; n - 2];

    // for j in 0..n {
    //     essential_u_vecs[j].set(j, 0, 1.0);
    //     essential_u_vecs[j].set_submatrix((j + 1)..m, 0..1, 
    //         work_matrix.get_submatrix((j + 1)..m, j..(j + 1))
    //     );
    //     work_matrix.set_submatrix((j + 1)..m, j..(j + 1), Matrix::new(m - j - 1, 1));

    //     if j < n - 2 {
    //         essential_v_vecs[j].set(j + 1, 0, 1.0);
            
    //         essential_v_vecs[j].set_submatrix((j + 2)..n, 0..1, 
    //             work_matrix.get_submatrix(j..(j + 1), (j + 2)..n).transpose()
    //         );
    //         work_matrix.set_submatrix(j..(j + 1), (j + 2)..n, Matrix::new(1, n - j - 2));
    //     }
    // }

    // println!("{:?}", essential_v_vecs[1]);
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
    use matrix_kit::dynamic::matrix::Matrix;

    use crate::math::svd::householder_bidiag;

    use super::golub_kahan_step;


    #[test]
    fn test_golub_kahan_step() {
        let diag = vec![1.0, 3.0, 0.5, -4.0];
        let updi = vec![2.0, -0.3, -0.1];

        let b = golub_kahan_step(diag, updi);
        println!("{:?}", b);
    }

    #[test]
    fn test_bidiagonalization() {
        let mut a = Matrix::random_normal(5, 4, 0.0, 1.0);
        println!("{:?}", a);
        let (u, v, b) = householder_bidiag(a.clone());
        println!("{:?}\n{:?}\n{:?}", u, v, b);

        println!("{:?}", u.transpose() * a * v);
    }

}