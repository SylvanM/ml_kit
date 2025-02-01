use std::{fs::File, io::Write, mem::transmute, process::exit};

use matrix_kit::dynamic::matrix::Matrix;

use crate::{math::activation::{ActivationFunctionIdentifier, AFI}, utility};


/// A shorthand for NeuralNet
pub type NN = NeuralNet;

/// A Neural Network (multi-layer perceptron) classifier, with a 
/// specified activation function
pub struct NeuralNet {

    /// The weights on edges between nodes in two adjacent layers 
    pub weights: Vec<Matrix<f64>>,

    /// The biases on nodes in each layer, except the first layer
    pub biases: Vec<Matrix<f64>>,

    /// A list of the activation functions used in each layer of this network
    pub activation_functions: Vec<AFI>,

}

impl NeuralNet {

    /// Checks that this NeuralNet is well-formed
    fn check_invariant(&self) {

        // Make sure we have the right amounts of each!
        debug_assert_eq!(self.weights.len(), self.biases.len());
        debug_assert_eq!(self.weights.len(), self.activation_functions.len());

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
    pub fn new(weights: Vec<Matrix<f64>>, biases: Vec<Matrix<f64>>, act_funcs: Vec<AFI>) -> NeuralNet {
        let nn = NeuralNet { weights, biases, activation_functions: act_funcs };
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
    pub fn compute_final_layer(&self, input: Matrix<f64>) -> Matrix<f64> {

        self.check_invariant();
        debug_assert_eq!(input.col_count(), 1);
        debug_assert_eq!(input.row_count(), self.weights[0].col_count());

        let mut current_output = input.clone();

        for layer in 0..self.weights.len() {
            current_output = self.weights[layer].clone() * current_output + self.biases[layer].clone();
            current_output.apply_to_all(&|x| self.activation_functions[layer].evaluate(x));
        }

        current_output
    }

    // MARK: File Utility

    /// Writes this neural net to a file
    pub fn write_to_file(&self, file: &mut File) {
        let mut header = vec![0u64 ; 2 * self.layer_count() + 1];
        header[0] = self.layer_count() as u64;

        for l in 0..self.layer_count() {
            if l == 0 {
                header[l + 1] = self.weights[0].col_count() as u64;
            } else {
                header[l + 1] = self.biases[l - 1].row_count() as u64;
            }
        }

        for l in 0..(self.layer_count() - 1) {
            header[self.layer_count() + 1 + l] = self.activation_functions[l].raw_value();
        }

        let header_bytes = utility::file_utility::u64s_to_bytes(header);

        match file.write(&header_bytes) {
            Ok(_) => {},
            Err(e) => {
                println!("Fatal Error Writing Header: {:?}", e);
                exit(-1);
            },
        }

        for weight in self.weights.clone() {

            match file.write(&utility::file_utility::floats_to_bytes(weight.as_vec())) {
                Ok(_) => {},
                Err(e) => {
                    println!("Fatal Error Writing Matrix {:?}Error: {:?}", weight, e);
                    exit(-1);
                },
            }

        }

        for bias in self.biases.clone() {

            match file.write(&utility::file_utility::floats_to_bytes(bias.as_vec())) {
                Ok(_) => {},
                Err(e) => {
                    panic!("Fatal Error Writing Biases {:?}Error: {:?}", bias, e);
                },
            }

        }


    }

    /// Reads a neural net from a file
    pub fn from_file(file: &mut File) -> NeuralNet {

    }

}

#[cfg(test)]
mod tests {
    use std::{fs::File, vec};

    use matrix_kit::dynamic::matrix::Matrix;

    use crate::math::activation::{ActivationFunctionIdentifier, AFI};

    use super::NeuralNet;

    // MARK: Basic Tests

    #[test]
    fn test_xor() {
        let first_weights = Matrix::<f64>::from_flatmap(2, 2, vec![1.0, -1.0, 1.0, -1.0]);
        let first_biases = Matrix::<f64>::from_flatmap(2, 1, vec![-0.5, 1.5]);
        let second_weights = Matrix::<f64>::from_flatmap(1, 2, vec![1.0, 1.0]);
        let second_biases = Matrix::<f64>::from_flatmap(1, 1, vec![-1.5]);

        let xor_nn = NeuralNet::new(vec![first_weights, second_weights], vec![first_biases, second_biases], vec![AFI::Sign, AFI::Sign]);

        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                println!("Input: {} ^ {}", x, y);
                println!("Output: {:?}", xor_nn.compute_final_layer(Matrix::<f64>::from_flatmap(2, 1, vec![x, y])));
            }
        }
    }

    // MARK: File I/O Tests

    #[test]
    fn test_file_io() {

        let first_weights = Matrix::<f64>::from_flatmap(2, 2, vec![1.0, -1.0, 1.0, -1.0]);
        let first_biases = Matrix::<f64>::from_flatmap(2, 1, vec![-0.5, 1.5]);
        let second_weights = Matrix::<f64>::from_flatmap(1, 2, vec![1.0, 1.0]);
        let second_biases = Matrix::<f64>::from_flatmap(1, 1, vec![-1.5]);

        let xor_nn = NeuralNet::new(vec![first_weights, second_weights], vec![first_biases, second_biases], vec![AFI::Sign, AFI::Sign]);

        // Write this down!
        let mut file = match File::create("tests/files/xor.mlk_nn") {
            Ok(f) => f,
            Err(e) => panic!("Error opening file: {:?}", e),
        };

        xor_nn.write_to_file(&mut file);

        // Now, attempt to read from the file, and make sure the nets are equal

        let decoded_nn = NeuralNet::from_file(&mut File::open("tests/files/xor.mlk_nn").unwrap());

        debug_assert_eq!(decoded_nn.layer_count(), xor_nn.layer_count());

        for l in 0..(decoded_nn.layer_count() - 1) {
            debug_assert_eq!(decoded_nn.weights[l], xor_nn.weights[l]);
        }

    }

}