use matrix_kit::dynamic::matrix::Matrix;

use crate::math::activation::ActivationFunction;

/// A shorthand for NeuralNet
pub type NN = NeuralNet;

/// A Neural Network (multi-layer perceptron) classifier, with a 
/// specified activation function
pub struct NeuralNet {

    /// The weights on edges between nodes in two adjacent layers 
    weights: Vec<Matrix<f64>>,

    /// The biases on nodes in each layer, except the first layer
    biases: Vec<Matrix<f64>>,

}

impl NeuralNet {

    /// Checks that this NeuralNet is well-formed
    fn check_invariant(&self) {

        // Make sure we have the right amounts of each!
        debug_assert_eq!(self.weights.len(), self.biases.len());

        // make sure all bias vectors are indeed vectors, and that they are 
        // of the right dimension, and that the weight matrix dimensions
        // line up

        debug_assert!(
            (0..self.weights.len()).all(|layer|
                self.biases[layer].col_count() == 1 &&
                self.biases[layer].row_count() == self.weights[layer].row_count() &&
                if layer == 0 { true } else {
                    self.weights[layer - 1].row_count() == self.weights[layer].col_count()
                }
            )
        )

    }

    // MARK: Constructors

    /// Creates a neural network with given weights and biases
    pub fn new(weights: Vec<Matrix<f64>>, biases: Vec<Matrix<f64>>) -> NeuralNet {
        let nn = NeuralNet { weights, biases };
        nn.check_invariant();
        nn
    }

    // MARK: Methods
    
    /// The amount of layers in this Neural Network, including the input layer
    pub fn layer_count(&self) -> usize {
        self.weights.len() + 1
    }

    /// Computes the output layer on a given input layer, using a specified
    /// activation function
    pub fn compute_final_layer<ActFunc: ActivationFunction>(&self, input: Matrix<f64>) -> Matrix<f64> {

        self.check_invariant();
        debug_assert_eq!(input.col_count(), 1);
        debug_assert_eq!(input.row_count(), self.weights[0].col_count());

        let mut current_output = input.clone();

        for layer in 0..self.weights.len() {
            current_output = self.weights[layer].clone() * current_output + self.biases[layer].clone();
            current_output.apply_to_all(&ActFunc::evaluate);
        }

        current_output
    }

}

#[cfg(test)]
mod tests {
    use std::vec;

    use matrix_kit::dynamic::matrix::Matrix;

    use crate::math::activation;

    use super::NeuralNet;


    #[test]
    fn test_xor() {
        let first_weights = Matrix::<f64>::from_flatmap(2, 2, vec![1.0, -1.0, 1.0, -1.0]);
        let first_biases = Matrix::<f64>::from_flatmap(2, 1, vec![-0.5, 1.5]);
        let second_weights = Matrix::<f64>::from_flatmap(1, 2, vec![1.0, 1.0]);
        let second_biases = Matrix::<f64>::from_flatmap(1, 1, vec![-1.5]);

        let xor_nn = NeuralNet::new(vec![first_weights, second_weights], vec![first_biases, second_biases]);

        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                println!("Input: {} ^ {}", x, y);
                println!("Output: {:?}", xor_nn.compute_final_layer::<activation::Sign>(Matrix::<f64>::from_flatmap(2, 1, vec![x, y])));
            }
        }
    }

}