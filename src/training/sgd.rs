//!
//! Stochastic Gradient Descent Trainer for Neural Networks
//!



use std::ops::{AddAssign, SubAssign};

use matrix_kit::dynamic::matrix::Matrix;
use rand_distr::Distribution;

use crate::{math::activation::AFI, models::neuralnet::NeuralNet};
use crate::math::loss::LFI;

use super::dataset::{DataItem, DataSet};
use super::learning_rate::GradientUpdateSchedule;


/// An SGD trainer that trains a neural network
pub struct SGDTrainer<DI: DataItem> {
    
    /// The dataset on which we train
    pub training_data_set: DataSet<DI>,

    /// The dataset on which we test
    pub testing_data_set: DataSet<DI>,

    /// The loss function used
    pub loss_function: LFI,

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
#[derive(Clone)]
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

// These are so we can compute averages of gradients!

impl AddAssign for NNGradient {
    fn add_assign(&mut self, rhs: Self) {
        for layer in 0..self.derivatives.weights.len() {
            self.derivatives.weights[layer] += rhs.derivatives.weights[layer].clone();
            self.derivatives.biases[layer] += rhs.derivatives.biases[layer].clone();
        }
    }
}

impl NNGradient {

    pub fn from_nn_shape(neuralnet: NeuralNet) -> NNGradient {
        let mut grad = NNGradient { derivatives: neuralnet };

        for layer in 0..grad.derivatives.weights.len() {
            grad.derivatives.weights[layer] = Matrix::from_index_def(grad.derivatives.weights[layer].row_count(), grad.derivatives.weights[layer].col_count(), &mut |_, _| 0.0);
            grad.derivatives.biases[layer] = Matrix::from_index_def(grad.derivatives.biases[layer].row_count(), 1, &mut |_, _| 0.0);
        }

        grad
    }

    pub fn norm(&self) -> f64 {
        let mut norm_squared = 0.0;

        for layer in 0..self.derivatives.weights.len() {
            norm_squared += self.derivatives.weights[layer].l2_norm_squared();
        }

        norm_squared.sqrt()
    }

    pub fn set_length(&mut self, length: f64) {
        let norm = self.norm();
        for layer in 0..self.derivatives.weights.len() {
            self.derivatives.weights[layer] /= norm;
            self.derivatives.weights[layer] *= length;
            self.derivatives.biases[layer] /= norm;
            self.derivatives.biases[layer] *= length;
        }
    }
}

impl<DI: DataItem> SGDTrainer<DI>  {

    pub fn new(training_data_set: DataSet<DI>, testing_data_set: DataSet<DI>, loss_function: LFI) -> SGDTrainer<DI> {
        SGDTrainer { training_data_set, testing_data_set, loss_function }
    }

    // MARK: Training

    /// Computes the gradient for a single training sample. The "gradient" is 
    /// represented as its own neural network, but a "derivative" neural network,
    /// if you will. (This just keeps everything organized)
    /// 
    /// TODO: Generalize this to think about more than just squared loss
    pub fn compute_gradient(&self, training_item: DI, neuralnet: &NeuralNet) -> NNGradient {
        let mut gradient = NNGradient { derivatives: neuralnet.clone() };

        let layers = neuralnet.layer_count() - 1; // The number of non-input layers. (Denotes as L in the writeup)

        let (z, a) = neuralnet.compute_raw_and_full_layers(training_item.input());
        let dot_sigma_z: Vec<Matrix<f64>> = (1..=layers)
            .map(
                |l| z[l].applying_to_all(
                    &|x| neuralnet.activation_functions[l - 1].derivative(x)
                )
            ).collect();

        let mut gradient_wrt_activations = a.clone(); // This is basically our DP table!

        // Base case of DP table, compute all dC/da for each activation in the final layer
        gradient_wrt_activations[layers] = self.loss_function.derivative(&a[layers], &training_item.correct_output()); 
        gradient.derivatives.biases[layers - 1] = dot_sigma_z[layers - 1].hadamard(gradient_wrt_activations[layers].clone());
        gradient.derivatives.weights[layers - 1] = gradient.derivatives.biases[layers - 1].clone() * a[layers - 1].transpose();

        // the rest now! I want the indices to actually match the indices in the writeup as closely as possible.

        for layer in (0..layers).rev() {
            gradient_wrt_activations[layer] = neuralnet.weights[layer].transpose().clone() * dot_sigma_z[layer].hadamard(gradient_wrt_activations[layer + 1].clone());
            gradient.derivatives.biases[layer] = dot_sigma_z[layer].hadamard(gradient_wrt_activations[layer + 1].clone());
            gradient.derivatives.weights[layer] = gradient.derivatives.biases[layer].clone() * a[layer].transpose().clone();
        }

        gradient
    }

    /// Performs a step of GD on a mini-batch of data, returning the size 
    /// of the gradient vector (before rescaling) so we can see how far from a local minimum we are.
    pub fn sgd_batch_step<GUS: GradientUpdateSchedule>(&self, batch: Vec<DI>, neuralnet: &mut NeuralNet, iteration: usize, gus: &mut GUS) -> f64 {
        // First, compute sum of gradients for all training items in the batch.
        let mut gradient = NNGradient::from_nn_shape(neuralnet.clone());

        for item in batch {
            gradient += self.compute_gradient(item, neuralnet);
        }

        let original_length = gradient.norm();

        // Now normalize that gradient!
        gus.update_gradient(&mut gradient, iteration);

        *neuralnet -= gradient;

        original_length
    }

    /// Runs Gradient Descent on this Data Set, outputting
    /// a neural network
    /// 
    /// * `neuralnet` - The neural network to train 
    /// * `lrs` - A learning rate schedule
    /// * `epochs` - the number of epochs to run
    /// * `batch_size` - the number of training items in each batch
    pub fn train_sgd<GUS: GradientUpdateSchedule>(&self, neuralnet: &mut NeuralNet, gus: &mut GUS, epochs: usize, batch_size: usize, verbose: bool) {
        // Repeat for all epochs!

        for epoch in 0..epochs {
            if verbose {
                println!("Training on epoch {}...", epoch);
            }

            let mut b = 0;
            for batch in self.training_data_set.all_minibatches(batch_size) {
                self.sgd_batch_step(batch, neuralnet, epoch * batch_size + b, gus);
                b += 1;
            }
        }

        if verbose {
            println!("Completed all epochs of training.");
        }
    }

    /// Generates a random neural network of a particular shape
    pub fn random_network(&self, shape: Vec<usize>, activation_functions: Vec<AFI>) -> NeuralNet {
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
        NeuralNet::new(weights, biases, activation_functions)
    }

    // MARK: Testing

    /// The average cost over all TESTING examples
    pub fn cost(&self, network: &NeuralNet) -> f64 {
        let mut average_cost = 0.0;

        let ds = &self.testing_data_set;

        for item in ds.data_items.clone() {
            let (x, y) = (item.input(), item.correct_output());
            let a = network.compute_final_layer(x);
            average_cost += self.loss_function.loss(&a, &y);
        }

        average_cost / (ds.data_items.len() as f64)
    }   

    /// The accuracy, as a percentage of testing items classified correctly
    pub fn accuracy(&self, network: &NeuralNet) -> f64 {
        let mut num_correct = 0;

        for item in self.testing_data_set.data_items.clone() {
            let (guess, _) = network.classify(item.input());

            if guess == item.label() {
                num_correct += 1;
            }
        }

        (num_correct as f64) / (self.testing_data_set.data_items.len() as f64)
    }

    /// Samples a few data items and prints to the screen the behavior 
    /// of the network
    pub fn display_behavior(&self, network: &NeuralNet, num_items: usize) {
        println!("Displaying network performance on {} testing items", num_items);

        for item in self.testing_data_set.random_sample(num_items) {
            println!("---Training Label: {} ---", item.name());
            println!("{:?}", item);
            println!("Network output: {:?}", network.classify(item.input()));
        }
        
        println!("--------------------");
        println!("Final cost: {}", self.cost(network));
        println!("Classification accuracy: {}", self.accuracy(network));
    }

}

#[cfg(test)]
mod sgd_tests {

    #[test]
    fn test_stuff() {

    }
}