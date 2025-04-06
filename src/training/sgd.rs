//!
//! Stochastic Gradient Descent Trainer for Neural Networks
//!


use std::{ops::SubAssign, os::macos::raw};

use matrix_kit::dynamic::matrix::Matrix;
use rand_distr::Distribution;

use crate::{math::activation::AFI, models::neuralnet::{self, NeuralNet, NN}};

use super::dataset::{DataItem, DataSet};


/// An SGD trainer that trains a neural network
pub struct SGDTrainer<DI: DataItem> {
    
    pub data_set: DataSet<DI>

}

/// A Gradient representation
/// 
/// We just use another neural net as the represntation, because this makes sure 
/// all the weight and bias gradients are well-orgaized. Each parameter in this 
/// gradient neural net maps to the derivative of the cost function with respect 
/// to the corresponding parameter of the original network.
/// 
/// Now, this let's us very intuitively "subtract" the gradient from our 
/// neural network, which in some sense is "applying" the step of 
/// gradient descent. This gradient is likely going to be normalized in 
/// some way.
pub struct NNGradient {
    pub derivatives: NeuralNet
}

impl SubAssign<NNGradient> for NeuralNet {
    fn sub_assign(&mut self, rhs: NNGradient) {
        for layer in 0..self.weights.len() {
            self.weights[layer] -= rhs.derivatives.weights[layer].clone();
            self.biases[layer] -= rhs.derivatives.biases[layer].clone();
        }
    }
}

impl<DI: DataItem> SGDTrainer<DI>  {

    pub fn new(data_set: DataSet<DI>) -> SGDTrainer<DI> {
        SGDTrainer { data_set }
    }

    // MARK: Training

    /// Computes the gradient for a single training sample. The "gradient" is 
    /// represented as its own neural network, but a "derivative" neural network,
    /// if you will. (This just keeps everything organized)
    /// 
    /// TODO: Generalize this to think about more than just squared loss
    pub fn compute_gradient(training_item: DI, neuralnet: NeuralNet) -> NNGradient {
        let mut gradient = NNGradient { derivatives: neuralnet.clone() };

        let (z, a) = neuralnet.compute_raw_and_full_layers(training_item.input());
        let mut gradient_wrt_activations = a.clone(); // This is basically our DP table!
        
        let layers = neuralnet.layer_count() - 1; // The number of non-input layers. (Denotes as L in the writeup)
        let layer_sizes = neuralnet.shape();


        // Base case of DP table, compute all dC/da for each activation in the final layer
        gradient_wrt_activations[layers - 1] = (a[layers - 1].clone() - training_item.correct_output()) * 2.0;
        gradient.derivatives.biases[layers - 1] = z[layers - 1].applying_to_all(&|x| neuralnet.activation_functions[layers - 1].derivative(x));
        gradient.derivatives.biases[layers - 1].hadamard(gradient_wrt_activations[layers - 1].clone());
        gradient.derivatives.weights[layers - 1] = gradient.derivatives.biases[layers - 1].clone() * a[layers - 2].transpose();

        // the rest now!

        for layer in 0..(layers - 1) {
            
        }

        gradient
    }

    /// Runs Stochastic Gradient Descent on this Data Set, outputting
    /// a neural network
    pub fn train_sgd(&self, shape: Vec<usize>, activation_functions: Vec<AFI>) -> NeuralNet {

        let mut rand_gen = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap(); // Tweak as needed!

        let weights = (1..shape.len()).map(
            |layer| {
                Matrix::from_index_def(shape[layer], shape[layer - 1], &mut |_, _| normal.sample(&mut rand_gen))
            }
        ).collect();
        let biases = (1..shape.len()).map(
            |layer| {
                Matrix::from_index_def(shape[layer], 1, &mut |_, _| normal.sample(&mut rand_gen))
            }
        ).collect();

        // Initialize a random neural network
        let mut neuralnet = NeuralNet::new(weights, biases, activation_functions);

        // For now, let's just play around! See what it outputs on a random thing
        let (_, all_layers) = neuralnet.compute_raw_and_full_layers(self.data_set.data_items[0].clone().input());

        println!("Layers after input: {:?}", all_layers);

        neuralnet
    }



}

#[cfg(test)]
mod sgd_tests {
    use crate::{math::activation::AFI, utility::mnist::mnist_utility::load_mnist};

    use super::SGDTrainer;


    #[test]
    fn test_network_stuff() {
        let dataset = load_mnist("train");
        let trainer = SGDTrainer::new(dataset);

        trainer.train_sgd(vec![784, 16, 16, 10], vec![AFI::Sigmoid, AFI::Sigmoid, AFI::Sigmoid]);
    }
}