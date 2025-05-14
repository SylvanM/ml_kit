//!
//! Mathematical Algorithms implementing Singular Value Decomposition
//! 
//! This uses the algorithms described in Golub and van Loan
//!

use core::f64;

use matrix_kit::dynamic::matrix::Matrix;

/// Algorithm 5.1.1 (Householder vector) from Golub and van Loan
fn house(x: Matrix<f64>) -> (Matrix<f64>, f64) {
    let x1 = x.get(0, 0);
    let x_2m = x.get_submatrix(1..x.row_count(), 0..1);
    let sigma = x_2m.inner_product(&x_2m);
    let mut v = Matrix::identity(1, 1);
    v.append_mat_bottom(x_2m);

    let beta;

    if sigma == 0.0 && x1 >= 0.0 {
        beta = 0.0;
    } else if sigma == 0.0 && x1 < 0.0 {
        beta = -2.0;
    } else {
        let mu = (x1.powf(2.0) + sigma).sqrt();
        
        if x1 <= 0.0 {
            v.set(0, 0, x1 - mu);
        } else {
            v.set(0, 0, 
                -sigma / (x1 + mu)
            );
        }

        let numerator = 2.0 * v.get(0, 0).powf(2.0);
        let denominator = sigma + v.get(0, 0).powf(2.0);
        beta = numerator / denominator;
        v /= v.get(0, 0);
    }

    (v, beta)
}

/// Algorithm 5.4.2 (Householder Bidiagonalization)
/// 
/// Takes in an m x n matrix, with m >= n, returns matrices U, V, and B such that 
/// 
/// U^T * A * V = B
///               0
fn householder_bidiag(
    matrix: Matrix<f64>
) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    let m = matrix.row_count();
    let n = matrix.col_count();
    let mut workmatrix = matrix.clone();

    let mut big_u = Matrix::identity(m, m);
    let mut big_v = Matrix::identity(n, n);

    for j in 0..n {
        let (v, _beta) = house(workmatrix.get_submatrix(j..m, j..(j + 1)));
        // workmatrix.set_submatrix(j..m, j..(j + 1), 
        //     workmatrix.get_submatrix(j..m, j..(j + 1)) - 
        //         (v.clone() * v.transpose() 
        //         * workmatrix.get_submatrix(j..m, j..(j + 1))
        //         * beta)
        // );
        // what if we just compute the ENTIRE householder matrix instead?

        let mut new_v = Matrix::new(m, 1);
        new_v.set_submatrix(j..m, 0..1, 
            v.clone()
        );

        let new_beta = 2.0 / new_v.l2_norm_squared();

        // println!("beta = {}, should be = {}", beta, 2.0 / new_v.l2_norm_squared());
        
        let householder_matrix = Matrix::identity(m, m) - (new_v.clone() * new_v.transpose() * new_beta);
        workmatrix = householder_matrix.transpose() * workmatrix;

        // Store the reflection we made to recover later
        // workmatrix.set_submatrix((j + 1)..m, j..(j + 1), 
        //     v.get_submatrix(1..(m - j), 0..1)
        // );
        
        // Now update U?
        // println!("Are we orthogonal? {:?}", householder_matrix.clone() * householder_matrix.transpose());
        big_u = householder_matrix * big_u;

        // Might have an issue if the matrix has 1 column, but why are we trynna
        // bidiagonalize that?
        if j < n - 2 {
            let (v, beta) = house(
                workmatrix.get_submatrix(j..(j + 1), (j + 1)..n).transpose()
            );
            // workmatrix.set_submatrix(j..m, (j + 1)..n,
            //     workmatrix.get_submatrix(j..m, (j + 1)..n) - (
            //         workmatrix.get_submatrix(j..m, (j + 1)..n) 
            //             * v.clone() * v.transpose() * beta
            //     )
            // );

            // we'll just compute the entire householder matrix!
            let mut new_v = Matrix::new(n, 1);
            new_v.set_submatrix((j + 1)..n, 0..1, 
                v.clone()
            );



            let householder_matrix = Matrix::identity(n, n) - (new_v.clone() * new_v.transpose() * beta);
            workmatrix = workmatrix * householder_matrix.clone();

            big_v = householder_matrix * big_v;

            // Store the reflection we made to recover later
            // workmatrix.set_submatrix(j..(j + 1), (j + 2)..n,
            //     v.get_submatrix(1..(n - j - 1), 0..1).transpose() // HUH?
            // );

        }
    }

    // Now, we wish to accumulate the matrices we've got!

    // Let's first compute big_u
    for j in (0..n).rev() {

        // let mut v = Matrix::new(m, 1);
        // v.set(j, 0, 1.0);
        // v.set_submatrix((j + 1)..m, 0..1, 
        //     workmatrix.get_submatrix((j + 1)..m, j..(j + 1))
        // );
        
        // let norm = workmatrix.get_submatrix((j + 1)..m, j..(j + 1))
        //                 .l2_norm_squared();

        // let beta = 2.0 / (1.0 + norm);
        // let new_householder_matrix = Matrix::identity(m, m) - (v.clone() * v.transpose() * beta);
        // big_u = new_householder_matrix * big_u;

        // zero out what we took
        workmatrix.set_submatrix((j + 1)..m, j..(j + 1), Matrix::new(m - j - 1, 1));
    }

    // Now, shall we compute big_v??? Man this will be scary!
    for j in (0..n - 2).rev() {
        // For each vector below the diagonal, we turn this into a 
        // householder rotation and apply it to U
        
        // let mut v = Matrix::new(j + 1, 1);
        // v.append_mat_bottom(Matrix::identity(1, 1));
        // v.append_mat_bottom(
        //     workmatrix.get_submatrix(j..(j + 1), (j + 2)..n).transpose()
        // );

        // let norm = workmatrix.get_submatrix(j..(j + 1), (j + 2)..n)
        //                     .l2_norm_squared();

        // let beta = 2.0 / (1.0 + norm);

        // let new_householder_matrix = Matrix::identity(n, n) - (v.clone() * v.transpose() * beta);
        
        // big_v = new_householder_matrix.clone() * big_v;

        // Zero out what we took
        workmatrix.set_submatrix(j..(j + 1), (j + 2)..n, Matrix::new(1, n - j - 2));
    }

    (big_u.transpose(), big_v.transpose(), workmatrix)
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

/// Sorts, in place, the rows, cols, and diagonal of the SVD of a matrix
/// so that it's in a useful form, and flips signs so that 
/// all singular values are positive.
fn format_svd(u: &mut Matrix<f64>, v: &mut Matrix<f64>, s: &mut Matrix<f64>) {

    let mut u_tups: Vec<(Matrix<f64>, f64)> = (0..u.col_count()).map(|c| 
        (
            u.get_submatrix(0..u.row_count(), c..(c + 1)),
            if c < s.col_count() {
                s.get(c, c).abs()
            } else {
                f64::NEG_INFINITY
            }
        )
    ).collect();

    let mut v_tups: Vec<(Matrix<f64>, f64)> = (0..v.col_count()).map(|c| 
        (
            v.get_submatrix(0..v.row_count(), c..(c + 1)) * s.get(c, c).signum(),
            s.get(c, c).abs()
        )
    ).collect();

    u_tups.sort_by(|(_, s1), (_, s2)| s2.total_cmp(s1));
    v_tups.sort_by(|(_, s1), (_, s2)| s2.total_cmp(s1));

    let u_cols = u_tups.iter().map(|(u_vec, _)| u_vec.clone()).collect();
    let v_cols = v_tups.iter().map(|(v_vec, _)| v_vec.clone()).collect();

    *u = Matrix::from_cols(u_cols);
    *v = Matrix::from_cols(v_cols);

    let mut diag: Vec<f64> = s.get_diagonal().iter().map(|sing| sing.abs()).collect();
    diag.sort_by(|a, b| b.abs().total_cmp(&a.abs()));

    *s = Matrix::from_diagonal(diag);

}

/// Computes the singular value decomposition of this matrix, `A`, returning 
/// a tuple `(U, V, S)` where `S` is a `Vec<f64>` of the singular values of
/// `A`, and
/// 
/// U^T * A * V = diag(S)
/// 
/// This is an implementation of Algorithm 8.6.2 from Golub and van Loan
fn svd_factorization(matrix: Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {

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


    // Make sure they are in a friendly, useful order
    format_svd(&mut u, &mut v, &mut b);
    b.append_mat_bottom(Matrix::new(m - n, n));
    (u, v, b)
}

/// Computes only the `r` most significant singular values and vectors,
/// used for compressing the data contained in matrix `A`.
/// 
/// The result will (U, V, S) so that S is diagonal and the columns of U and V 
/// are orthogonal (though the matrices themselves are not square.)
/// 
/// S will be an r-length Vec<f64>
/// U will be m x r
/// V will be n x r
pub fn compressed_svd(
    matrix: Matrix<f64>, r: usize
) -> (Matrix<f64>, Matrix<f64>, Vec<f64>) {

    let (u, v, s) = svd_factorization(matrix);

    let u_r = u.get_submatrix(0..u.row_count(), 0..r);
    let v_r = v.get_submatrix(0..v.row_count(), 0..r);
    let s_r = s.get_submatrix(0..r, 0..r);

    (u_r, v_r, s_r.get_diagonal())
}

/// Computes the Singular Value Decomposition of an `m` by `n` matrix `A`,
/// returning the orthogonal matrices `U` and `V`, and the diagonal matrix
/// `S`. The columns of `V` are the right singular vectors of `A`, and the 
/// columns of `U` are the left singular vectors. The diagonal of `S` 
/// is the singular values of `A`.
pub fn svd(matrix: Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    let m = matrix.row_count();
    let n = matrix.col_count();

    if m >= n {
        svd_factorization(matrix)
    } else {
        let transposed = matrix.transpose();
        let (u, v, s) = svd_factorization(transposed);
        (v, u, s.transpose())
    }
}

#[cfg(test)]
mod svd_math_tests {
    use matrix_kit::dynamic::matrix::Matrix;
    use rand::Rng;
    use super::compressed_svd;
    use super::house;
    use super::householder_bidiag;
    use super::svd;

    fn matrices_close(a: &Matrix<f64>, b: &Matrix<f64>) -> bool {
        if a.row_count() != b.row_count() || a.col_count() != b.col_count() {
            return false;
        } 
        
        for r in 0..a.row_count() {
            for c in 0..a.col_count() {
                if (a.get(r, c) - b.get(r, c)).abs() > 1e-5 {
                    return false;
                }
            }
        }

        return true;
    }

    #[test]
    fn test_house() {
        let x = Matrix::from_flatmap(2, 1, vec![-1.0, 1.0]);
        let (v, beta) = house(x);
        println!("House: {:?}, {:?}", beta, v);
    }

    #[test]
    fn test_bidiagonalization() {

        for _ in 1..=1000 {
            let mut rng = rand::rng();

            let n = rng.random_range(2..=40);
            let m = rng.random_range(n..100);

            let mut a = Matrix::<f64>::random_normal(m, n, 0.0, 1.0);
            // let a = Matrix::from_flatmap(2, 2, vec![
            //     -1.0, -1.0, -1.0, 1.0
            // ]);
            a.apply_to_all(&|x: f64| x.abs());

            let (u, v, b) = householder_bidiag(a.clone());
            
            // println!("U [{} x {}]: {:?}", u.row_count(), u.col_count(), u.clone());
            // println!("S [{} x {}]: {:?}", b.row_count(), b.col_count(), b.clone());
            // println!("V [{} x {}]: {:?}", v.row_count(), v.col_count(), v.clone());
            // make sure u and v are actually orthogonal

            if !matrices_close(&(u.transpose() * u.clone()), &Matrix::identity(u.row_count(), u.col_count())) && 
               !matrices_close(&(v.transpose() * v.clone()), &Matrix::identity(v.row_count(), v.col_count())){
                println!("U and V are not orthogonal.");
                println!("A:{:?}", a);
                println!("U: {:?}", u.clone());
                println!("U^T U:{:?}", u.transpose() * u.clone());
                println!("V^T V:{:?}", v.transpose() * v.clone());
                panic!("U and V are not orthogonal");
            }

            if !matrices_close(&(u.transpose() * a.clone() * v.clone()), &b) {
                println!("Improper factorization");
                println!("A:{:?}", a);
                println!("U: {:?}", u.clone());
                println!("U^T U:{:?}", u.transpose() * u.clone());
                println!("V^T V:{:?}", v.transpose() * v.clone());
                println!("U^T * A * V: {:?}", u.transpose() * a.clone() * v.clone());
                println!("U * A * V^T: {:?}", u.clone() * a.clone() * v.transpose());
                println!("B: {:?}", b);
                panic!("Wrong factorization");
            }

            if !matrices_close(&(u.clone() * b.clone() * v.transpose()), &a) {
                println!("Improper factorization");
                println!("A:{:?}", a);
                println!("U: {:?}", u.clone());
                println!("U^T U:{:?}", u.transpose() * u.clone());
                println!("V^T V:{:?}", v.transpose() * v.clone());
                println!("U^T * A * V: {:?}", u.transpose() * a.clone() * v);
                println!("B: {:?}", b);
                panic!("Wrong factorization");
            }
        }
    }

    #[test]
    fn test_svd() {
        let a = Matrix::from_flatmap(3, 4, vec![
            0.0, 2.0, 4.0, 6.0,
            1.0, 3.0, 5.0, 7.0,
            -1.0, 2.0, -3.0, -4.0,
        ]);

        let (u, v, s) = svd(a.clone());

        println!("Orthogonal check!");
        println!("{:?}", u.transpose() * u.clone());
        println!("{:?}", v.transpose() * v.clone());

        println!("Singular values?");
        println!("{:?}", s);

        println!("Can we get diagonal?");
        println!("{:?}", u.transpose() * a.clone() * v.clone());
        
        println!("Can we recover?");
        println!("{:?}", u.clone() * s.clone() * v.transpose());

        // for _ in 1..=1 {
        //     let mut rng = rand::rng();

        //     let n = rng.random_range(2..=40);
        //     let m = rng.random_range(2..=40);

        //     println!("SVD-ing [{} x {}]", m, n);

        //     let a = Matrix::random_normal(m, n, 0.0, 1.0);

        //     let (u, v, s) = svd(a.clone());

        //     let alleged_a = u.clone() * s.clone() * v.transpose();
        //     let alleged_s = u.transpose() * a.clone() * v.clone();

        //     assert!(matrices_close(&alleged_a, &a));
        //     assert!(matrices_close(&alleged_s, &s));
        // }
    }

    #[test]
    fn test_svd_compression() {
        let a = Matrix::from_flatmap(4, 3, vec![
            0.0, 2.0, 4.0, 6.0,
            1.0, 3.0, 5.0, 7.0,
            -1.0, 2.0, -3.0, -4.0,
        ]);

        let (u, v, s) = compressed_svd(a.clone(), 2);

        println!("Orthogonal check!");
        println!("{:?}", u.transpose() * u.clone());
        println!("{:?}", v.transpose() * v.clone());

        println!("Singular values?");
        println!("{:?}", s);

        println!("Can we get diagonal?");
        println!("{:?}", u.transpose() * a.clone() * v.clone());
        
        println!("Can we recover?");
        let s_matrix = Matrix::from_index_def(u.col_count(), v.col_count(), &mut |r, c| if r == c {
            s[r] 
        }else {
                0.0
            });
        println!("{:?}", u.clone() * s_matrix * v.transpose());
    }
}