use matrix_kit::dynamic::matrix::Matrix;

pub type LFI = LossFunctionIdentifier;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LossFunctionIdentifier {
    Squared,
    // Additional loss functions (e.g., CrossEntropy) can be added here.
}

const LOSS_LOOKUP_TABLE: [LossFunctionIdentifier; 1] = [
    LossFunctionIdentifier::Squared,
];

impl LossFunctionIdentifier {
    pub fn from_int(n: u64) -> LossFunctionIdentifier {
        LOSS_LOOKUP_TABLE[n as usize]
    }

    pub fn raw_value(self) -> u64 {
        match self {
            LossFunctionIdentifier::Squared => 0,
        }
    }

    pub fn loss(self, prediction: &matrix_kit::dynamic::matrix::Matrix<f64>, target: &matrix_kit::dynamic::matrix::Matrix<f64>) -> f64 {
        match self {
            LossFunctionIdentifier::Squared => SquaredLoss {}.loss(prediction, target),
        }
    }

    pub fn derivative(self, prediction: &matrix_kit::dynamic::matrix::Matrix<f64>, target: &matrix_kit::dynamic::matrix::Matrix<f64>) -> Matrix<f64> {
        match self {
            LossFunctionIdentifier::Squared => SquaredLoss {}.derivative(prediction, target),
        }
    }
}


pub trait LossFunction { 

	fn loss(&self, prediction: &Matrix<f64>, target: &Matrix<f64>) -> f64; 

	fn derivative(&self, prediction: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64>; 

}

pub struct SquaredLoss; 

impl LossFunction for SquaredLoss { 

	fn loss(&self, prediction: &Matrix<f64>, target: &Matrix<f64>) ->f64 { 
		((prediction.clone()) - target.clone()).l2_norm_squared()
	}

	fn derivative(&self, prediction: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
        (prediction.clone() - target.clone()) * 2.0
    }
}