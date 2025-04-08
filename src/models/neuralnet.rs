use core::panic;
use std::{fmt::Debug, fs::File, io::{Read, Write}, os::macos::raw, process::exit, vec};

use matrix_kit::dynamic::matrix::Matrix;

use crate::{math::activation::AFI, training::dataset::DataItem, utility};


/// A shorthand for NeuralNet
pub type NN = NeuralNet;

/// A Neural Network (multi-layer perceptron) classifier, with a 
/// specified activation function
#[derive(Clone)]
pub struct NeuralNet {

    /// The weights on edges between nodes in two adjacent layers
    pub weights: Vec<Matrix<f64>>,

    /// The biases on nodes in each layer, except the first layer
    pub biases: Vec<Matrix<f64>>,

    /// A list of the activation functions used in each layer of this network
    pub activation_functions: Vec<AFI>,

}

impl Debug for NeuralNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralNet")
            .field("weights", &self.weights)
            .field("biases", &self.biases)
            .field("activation_functions", &self.activation_functions)
            .finish()
    }
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
                self.biases[layer].row_count() == self.weights[layer] .row_count() 
                && if layer == 0 { true } else {
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

    /// The shape of this neural network, with the 0th element 
    /// of this array representing the size of the input
    pub fn shape(&self) -> Vec<usize> {
        let mut noninput_shape: Vec<usize> = self.biases.iter().map(|bias| bias.row_count()).collect();
        let mut shape = vec![self.weights[0].col_count()];
        shape.append(&mut noninput_shape);
        shape
    }

    /// Computes the output layer on a given input layer
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

    /// Computes compute the activation of all layers of this network, WITHOUT 
    /// the activation funtion being applied. (though it is applied to compute 
    /// layer layers)
    pub fn compute_raw_layers(&self, input: Matrix<f64>) -> Vec<Matrix<f64>> {

        self.check_invariant();
        debug_assert_eq!(input.col_count(), 1);
        debug_assert_eq!(input.row_count(), self.weights[0].col_count());

        let mut layers = Vec::new();

        let mut current_output = input.clone();

        for layer in 0..self.weights.len() {
            current_output = self.weights[layer].clone() * current_output + self.biases[layer].clone();
            layers.push(current_output.clone());
            current_output.apply_to_all(&|x| self.activation_functions[layer].evaluate(x));
        }

        layers
    }

    /// Computes all raw activations as well as activations after the activation
    /// function has been applied. (raw, full)
    pub fn compute_raw_and_full_layers(&self, input: Matrix<f64>) -> (Vec<Matrix<f64>>, Vec<Matrix<f64>>) {
        let raw_layers = self.compute_raw_layers(input.clone());

        let mut full_layers: Vec<Matrix<f64>> = (0..(self.layer_count() - 1)).map(
            |l| raw_layers[l]
            .applying_to_all(&|x| self.activation_functions[l].evaluate(x))
        ).collect();

        full_layers.insert(0, input);

        (raw_layers, full_layers)
    }

    // MARK: File Utility

    /// Writes this neural net to a file
    pub fn write_to_file(&self, file: &mut File) {

        self.check_invariant(); // Wouldn't want to store a malformed neural net!

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
        // read first 8 bytes to see the size
        let mut lc_buffer = [0u8 ; 8];

        match file.read(&mut lc_buffer) {
            Ok(_) => {},
            Err(e) => panic!("Error reading layer cound: {:?}", e)
        }

        let layer_count = utility::file_utility::bytes_to_u64s(lc_buffer.to_vec())[0] as usize;

        // buffer for layer sizes and activation function ID's
        let mut lsa_buffer = vec![0u8 ; (layer_count + layer_count) * 8];

        match file.read(&mut lsa_buffer) {
            Ok(_) => {},
            Err(e) => panic!("Error reading layer sizes: {:?}", e)
        }

        let layer_sizes_and_acts = utility::file_utility::bytes_to_u64s(lsa_buffer);

        let mut weights = Vec::new();

        // go through and get read some weight matrices!
        for layer in 0..(layer_count - 1) {
            let cols = layer_sizes_and_acts[layer] as usize;
            let rows = layer_sizes_and_acts[layer + 1] as usize;
            let mut mat_buff = vec![0u8; rows * cols * 8];

            match file.read(&mut mat_buff) {
                Ok(_) => {},
                Err(e) => panic!("Error reading weight matrix {}: {:?}", layer, e)
            }

            let flatmap = utility::file_utility::bytes_to_floats(mat_buff);

            weights.push(Matrix::from_flatmap(rows, cols, flatmap));
        }

        let mut biases = Vec::new();

        // go through and get read some bias vectors!
        for layer in 0..(layer_count - 1) {
            let rows = layer_sizes_and_acts[layer + 1] as usize;
            let mut vec_buff = vec![0u8; rows * 8];

            match file.read(&mut vec_buff) {
                Ok(_) => {},
                Err(e) => panic!("Error reading bias vector {}: {:?}", layer, e)
            }

            let flatmap = utility::file_utility::bytes_to_floats(vec_buff);

            biases.push(Matrix::from_flatmap(rows, 1, flatmap));
        }

        let activation_functions = layer_sizes_and_acts[layer_count..(2 * layer_count - 1)]
            .iter()
            .map(|id| AFI::from_int(*id))
            .collect();

        NeuralNet::new(weights, biases, activation_functions)
        
    }

}

#[cfg(test)]
mod tests {
    use std::{fs::File, vec};

    use matrix_kit::dynamic::matrix::Matrix;
    use crate::math::activation::AFI;
    use super::NeuralNet;

    // MARK: File I/O Tests

    #[test]
    fn test_file_io() {

        let first_weights   = Matrix::<f64>::from_flatmap(2, 2, vec![1.0, -1.0, 1.0, -1.0]);
        let first_biases    = Matrix::<f64>::from_flatmap(2, 1, vec![-0.5, 1.5]);
        let second_weights  = Matrix::<f64>::from_flatmap(1, 2, vec![1.0, 1.0]);
        let second_biases   = Matrix::<f64>::from_flatmap(1, 1, vec![-1.5]);

        let xor_nn = NeuralNet::new(
            vec![first_weights, second_weights], 
            vec![first_biases, second_biases], 
            vec![AFI::Sign, AFI::Step]
        );

        // Write this down!
        let mut file = match File::create("testing/files/xor.mlk_nn") {
            Ok(f) => f,
            Err(e) => panic!("Error opening file: {:?}", e),
        };

        xor_nn.write_to_file(&mut file);

        // Now, attempt to read from the file, and make sure the nets are equal

        let decoded_nn = NeuralNet::from_file(&mut File::open("testing/files/xor.mlk_nn").unwrap());

        debug_assert_eq!(decoded_nn.layer_count(), xor_nn.layer_count());

        for l in 0..(decoded_nn.layer_count() - 1) {
            debug_assert_eq!(decoded_nn.weights[l], xor_nn.weights[l]);
            debug_assert_eq!(decoded_nn.biases[l], xor_nn.biases[l]);
            debug_assert_eq!(decoded_nn.activation_functions[l], xor_nn.activation_functions[l]);
        }

    }

    #[test]
    fn test_xor() {
        let mut file = match File::open("testing/files/xor.mlk_nn") {
            Ok(f) => f,
            Err(e) => panic!("Error opening file: {:?}", e),
        };
        let xor_nn = NeuralNet::from_file(&mut file);

        for x in [0, 1] {
            for y in [0, 1] {
                let output = xor_nn.compute_final_layer(Matrix::<f64>::from_flatmap(2, 1, vec![x as f64, y as f64])).get(0, 0) as u64;
                println!("Output: {} ^ {} = {:?}", x, y, output);

                debug_assert_eq!(x ^ y, output)
            }
        }
    }

}