//!
//! A collection of functions for learning rate schedules
//!

use std::usize;

use rand_distr::Iter;

use super::sgd::NNGradient;

/// A generalized version of a learning rate schedule, this changes the gradient 
/// at each step by either scaling it by some learning rate (a function 
/// of the iteration, or something else) or performing some other transformation.
/// This happens to the *unnormalized gradient*.
pub trait GradientUpdateSchedule {
    fn update_gradient(&mut self, gradient: &mut NNGradient, iteration: usize);
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
    fn update_gradient(&mut self, gradient: &mut NNGradient, _iteration: usize) {
        gradient.set_length(self.rate);
    }

}

/// A Time-based decayed learning rate
pub struct TimeDecay {
    current_rate: f64,
    decay: f64,
    iteration: usize,
}

impl TimeDecay {
    /// Creates a new TimeDecay rate schedule
    pub fn new(current_rate: f64, decay: f64) -> TimeDecay {
        TimeDecay { current_rate, decay, iteration: 0 }
    }
}

impl GradientUpdateSchedule for TimeDecay {

    /// Normalizes the gradient to have a fixed length
    fn update_gradient(&mut self, gradient: &mut NNGradient, iteration: usize) {
        debug_assert_eq!(iteration, self.iteration + 1, "Rates should be called in the right order!");

        self.current_rate = self.current_rate / (1.0 + self.decay * (iteration as f64));
        self.iteration += 1;
        gradient.set_length(self.current_rate);
    }
}