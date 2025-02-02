/// Don't want to keep typing this mouthful
pub type AFI = ActivationFunctionIdentifier;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActivationFunctionIdentifier {
    Identity,
    Sign,
    ReLu,
    Sigmoid,
    Step,
}

const IDEN_LOOKUP_TABLE: [ActivationFunctionIdentifier ; 5] = [ 
    ActivationFunctionIdentifier::Identity,
    ActivationFunctionIdentifier::Sign,
    ActivationFunctionIdentifier::ReLu,
    ActivationFunctionIdentifier::Sigmoid,
    ActivationFunctionIdentifier::Step,
];

impl ActivationFunctionIdentifier {

    /// Creates a new AFI from a numerical value 
    pub fn from_int(n: u64) -> ActivationFunctionIdentifier {
        IDEN_LOOKUP_TABLE[n as usize]
    }

    /// Returns the raw value of this AFI
    pub fn raw_value(self) -> u64 {
        match self {
            ActivationFunctionIdentifier::Identity  => 0,
            ActivationFunctionIdentifier::Sign      => 1,
            ActivationFunctionIdentifier::ReLu      => 2,
            ActivationFunctionIdentifier::Sigmoid   => 3,
            ActivationFunctionIdentifier::Step      => 4,
        }
    }

    /// Evaluates this activation function on a certain value, `x`
    pub fn evaluate(self, x: f64) -> f64 {
        match self {
            ActivationFunctionIdentifier::Identity  => (Identity {}).evaluate(x),
            ActivationFunctionIdentifier::Sign      => (Sign {}).evaluate(x),
            ActivationFunctionIdentifier::ReLu      => (ReLu {}).evaluate(x),
            ActivationFunctionIdentifier::Sigmoid   => (Sigmoid {}).evaluate(x),
            ActivationFunctionIdentifier::Step      => (Step {}).evaluate(x),
        }
    }

    /// Computes the derivative of this activation function at a certain value, `x`
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            ActivationFunctionIdentifier::Identity  => (Identity {}).derivative(x),
            ActivationFunctionIdentifier::Sign      => (Sign {}).derivative(x),
            ActivationFunctionIdentifier::ReLu      => (ReLu {}).derivative(x),
            ActivationFunctionIdentifier::Sigmoid   => (Sigmoid {}).derivative(x),
            ActivationFunctionIdentifier::Step      => (Step {}).derivative(x),
        }
    }

}

/// A function ``squishing'' the activation of a neuron 
/// 
/// 3Blue1Brown calls this the ``squishification'' function.
trait ActivationFunction {
    
    /// Evaluates the activation function at a given value
    fn evaluate(&self, x: f64) -> f64;

    /// Evaluates the derivative of this function at a given value
    fn derivative(&self, x: f64) -> f64;

}

// MARK: Default Implementations

/// The Identity function, which does nothing
struct Identity {}
impl ActivationFunction for Identity {

    fn evaluate(&self, x: f64) -> f64 {
        x
    }

    fn derivative(&self, _x: f64) -> f64 {
        1.0
    }
}

struct Sign {}
impl ActivationFunction for Sign {
    fn evaluate(&self, x: f64) -> f64 {
        x.signum()
    }

    fn derivative(&self, _x: f64) -> f64 {
        0.0
    }
}

/// The Rectified Linear Unit function
struct ReLu {}
impl ActivationFunction for ReLu {
    fn evaluate(&self, x: f64) -> f64 {
        if x >= 0.0 { x } else { 0.0 }
    }

    fn derivative(&self, x: f64) -> f64 {
        if x == 0.0 { f64::NAN } // We might need to change this if it causes errors
        else if x > 0.0 { 1.0 }
        else { 0.0 }
    }
}

/// The Sigmoid Function
struct Sigmoid {}
impl ActivationFunction for Sigmoid {
    fn evaluate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f64) -> f64 {
        self.evaluate(x) * (1.0 - self.evaluate(x))
    }
}

/// The Step function Function
struct Step {}
impl ActivationFunction for Step {
    fn evaluate(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            1.0
        }
    }

    fn derivative(&self, _x: f64) -> f64 {
        0.0
    }
}