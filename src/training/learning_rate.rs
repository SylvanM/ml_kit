//!
//! A collection of functions for learning rate schedules
//!
use std::usize;

use matrix_kit::dynamic::matrix::Matrix;
use rand_distr::num_traits::Pow;

use super::sgd::NNGradient;

/// A generalized version of a learning rate schedule, this changes the gradient
/// at each step by either scaling it by some learning rate (a function
/// of the iteration, or something else) or performing some other transformation.
/// This happens to the *unnormalized gradient*.
pub trait GradientUpdateSchedule {
    fn next_gradient(&mut self, gradient: &mut NNGradient);
}

// MARK: Implementations

/// A fixed learning rate
pub struct FixedLR {
    rate: f64,
}

impl FixedLR {
    /// Creates a new Fixed Learning rate schedule with the provided rate
    pub fn new(rate: f64) -> FixedLR {
        FixedLR { rate }
    }
}

impl GradientUpdateSchedule for FixedLR {
    /// Normalizes the gradient to have a fixed length
    fn next_gradient(&mut self, gradient: &mut NNGradient) {
        gradient.set_length(self.rate);
    }
}

// Epsilon value to fill diag instead of 0s
const EPS: f64 = 1e-8;

/// The AdaGrad update schedule, which stores the learning rate, and the
/// diagonal of the sum of the outer-product of the gradients for each step
pub struct AdaGrad {
    rate: f64,

    /// The diagonal of the sum of all g*g^T, where g is the gradient vector
    outer_product_diag: Matrix<f64>,
}

impl AdaGrad {
    /// Creates a new AdaGrad instance with a certain learning rate
    pub fn new(rate: f64, diag_size: usize) -> AdaGrad {
        let mut outer = Matrix::new(diag_size, 1);
        outer.apply_to_all(&|_| EPS);
        AdaGrad {
            rate,
            outer_product_diag: outer,
        }
    }
    pub fn decay(&mut self, decay_factor: f64) {
        self.outer_product_diag *= decay_factor;
    }
}

impl GradientUpdateSchedule for AdaGrad {
    fn next_gradient(&mut self, gradient: &mut NNGradient) {
        let grad = gradient.as_vec();

        self.outer_product_diag += grad.applying_to_all(&|g_d: f64| g_d.pow(2));

        let _scales = self
            .outer_product_diag
            .applying_to_all(&|g| self.rate / ((g + EPS).sqrt()));

        let adjustment = self
            .outer_product_diag
            .applying_to_all(&mut |z| {
                if z == 0.0 {
                    0.0
                } else {
                    self.rate / z.sqrt()
                }
            })
            .hadamard(grad);

        *gradient = NNGradient::from_vec(adjustment, gradient.derivatives.shape());
    }
}
