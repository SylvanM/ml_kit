//!
//! A collection of functions for learning rate schedules
//!
use std::usize;

use matrix_kit::dynamic::matrix::Matrix;
use rand_distr::num_traits::Pow;

use super::sgd::{NNGradient, CNNGradient};

/// A generalized version of a learning rate schedule, this changes the gradient
/// at each step by either scaling it by some learning rate (a function
/// of the iteration, or something else) or performing some other transformation.
/// This happens to the *unnormalized gradient*. 
/// Made trait generic, not hardcoded to NNGradient
pub trait GradientUpdateSchedule<G> {
    fn next_gradient(&mut self, gradient: &mut G);
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

impl GradientUpdateSchedule<NNGradient> for FixedLR {
    /// Normalizes the nn gradient to have a fixed length
    fn next_gradient(&mut self, gradient: &mut NNGradient) {
        gradient.set_length(self.rate);
    }
}

impl GradientUpdateSchedule<CNNGradient> for FixedLR {
    /// Normalizes the cnn gradient to have a fixed length
    fn next_gradient(&mut self, gradient: &mut CNNGradient) {
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

impl GradientUpdateSchedule<NNGradient> for AdaGrad {
    fn next_gradient(&mut self, gradient: &mut NNGradient) {
        let grad = gradient.as_vec();

        self.outer_product_diag += grad.applying_to_all(&|g_d: f64| g_d.pow(2));

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

impl GradientUpdateSchedule<CNNGradient> for AdaGrad {
    fn next_gradient(&mut self, gradient: &mut CNNGradient) {
        // flattens 
        let mut grad_vec = Vec::new();
        for layer in &gradient.derivatives {
            match layer {
                Layer::Conv(conv) => {
                    for filter in &conv.filters {
                        for depth in filter {
                            grad_vec.extend(depth.as_vec());
                        }
                    }
                    grad_vec.extend(conv.biases.as_vec());
                }
                Layer::Full(full) => {
                    grad_vec.extend(full.weights.as_vec());
                    grad_vec.extend(full.biases.as_vec());
                }
                _ => {} 
            }
        }
        
        let grad_matrix = Matrix::from_flatmap(grad_vec.len(), 1, grad_vec);
        
        self.outer_product_diag += grad_matrix.applying_to_all(&|g_d: f64| g_d.pow(2));
        
        let adjustment = self
            .outer_product_diag
            .applying_to_all(&mut |z| {
                if z == 0.0 {
                    0.0
                } else {
                    self.rate / z.sqrt()
                }
            })
            .hadamard(&grad_matrix);
            
        // reshapes back to cnn
        let mut i = 0;
        for layer in &mut gradient.derivatives {
            match layer {
                Layer::Conv(conv) => {
                    for filter in &mut conv.filters {
                        for depth in filter {
                            let size = depth.row_count() * depth.col_count();
                            let mut new_depth = Matrix::new(depth.row_count(), depth.col_count());
                            for r in 0..depth.row_count() {
                                for c in 0..depth.col_count() {
                                    new_depth.set(r, c, adjustment.get(i, 0));
                                    i += 1;
                                }
                            }
                            *depth = new_depth;
                        }
                    }
                    let bias_size = conv.biases.row_count();
                    let mut new_biases = Matrix::new(bias_size, 1);
                    for b in 0..bias_size {
                        new_biases.set(b, 0, adjustment.get(i, 0));
                        i += 1;
                    }
                    conv.biases = new_biases;
                }
                Layer::Full(full) => {
                           let weight_size = full.weights.row_count() * full.weights.col_count();
                    let mut new_weights = Matrix::new(full.weights.row_count(), full.weights.col_count());
                    for r in 0..full.weights.row_count() {
                        for c in 0..full.weights.col_count() {
                            new_weights.set(r, c, adjustment.get(i, 0));
                            i += 1;
                        }
                    }
                    full.weights = new_weights;
                    

                    let bias_size = full.biases.row_count();
                    let mut new_biases = Matrix::new(bias_size, 1);
                    for b in 0..bias_size {
                        new_biases.set(b, 0, adjustment.get(i, 0));
                        i += 1;
                    }
                    full.biases = new_biases;
                }
                _ => {} /
            }
        }
    }
}
