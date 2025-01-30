/// A function ``squishing'' the activation of a neuron 
/// 
/// 3Blue1Brown calls this the ``squishification'' function.
pub trait ActivationFunction {
    
    /// Evaluates the activation function at a given value
    fn evaluate(x: f64) -> f64;

    /// Evaluates the derivative of this function at a given value
    fn derivative(x: f64) -> f64;

}

// MARK: Default Implementations

/// The Identity function, which does nothing
pub struct Identity {}
impl ActivationFunction for Identity {
    fn evaluate(x: f64) -> f64 {
        x
    }

    fn derivative(_x: f64) -> f64 {
        1.0
    }
}

pub struct Sign {}
impl ActivationFunction for Sign {
    fn evaluate(x: f64) -> f64 {
        x.signum()
    }

    fn derivative(_x: f64) -> f64 {
        0.0
    }
}

/// The Rectified Linear Unit function
pub struct ReLu {}
impl ActivationFunction for ReLu {
    fn evaluate(x: f64) -> f64 {
        if x >= 0.0 { x } else { 0.0 }
    }

    fn derivative(x: f64) -> f64 {
        if x == 0.0 { f64::NAN } // We might need to change this if it causes errors
        else if x > 0.0 { 1.0 }
        else { 0.0 }
    }
}

/// The Sigmoid Function
pub struct Sigmoid {}
impl ActivationFunction for Sigmoid {
    fn evaluate(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(x: f64) -> f64 {
        Self::evaluate(x) * (1.0 - Self::evaluate(x))
    }
}