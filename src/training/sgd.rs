//!
//! Stochastic Gradient Descent Trainer for Neural Networks
//!

use std::{
    alloc::Layout,
    ops::{AddAssign, DivAssign, SubAssign},
    os::macos::raw,
};

use matrix_kit::dynamic::matrix::Matrix;
use rand::distr::weighted;
use rand_distr::Distribution;
use rand_distr::Uniform;

use crate::{
    math::activation::AFI,
    models::neuralnet::{self, NeuralNet, NN},
};

use super::dataset::{DataItem, DataSet};

/// An SGD trainer that trains a neural network
pub struct SGDTrainer<DI: DataItem> {
    pub data_set: DataSet<DI>,
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
    pub derivatives: NeuralNet,
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
        let mut grad = NNGradient {
            derivatives: neuralnet,
        };

        for layer in 0..grad.derivatives.weights.len() {
            grad.derivatives.weights[layer] = Matrix::from_index_def(
                grad.derivatives.weights[layer].row_count(),
                grad.derivatives.weights[layer].col_count(),
                &mut |_, _| 0.0,
            );
            grad.derivatives.biases[layer] = Matrix::from_index_def(
                grad.derivatives.biases[layer].row_count(),
                1,
                &mut |_, _| 0.0,
            );
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

impl<DI: DataItem> SGDTrainer<DI> {
    pub fn new(data_set: DataSet<DI>) -> SGDTrainer<DI> {
        SGDTrainer { data_set }
    }

    // MARK: Training

    /// Computes the gradient for a single training sample. The "gradient" is
    /// represented as its own neural network, but a "derivative" neural network,
    /// if you will. (This just keeps everything organized)
    ///
    /// TODO: Generalize this to think about more than just squared loss
    pub fn compute_gradient(training_item: DI, neuralnet: &NeuralNet) -> NNGradient {
        let mut gradient = NNGradient {
            derivatives: neuralnet.clone(),
        };

        let layers = neuralnet.layer_count() - 1; // The number of non-input layers. (Denotes as L in the writeup)

        let (z, a) = neuralnet.compute_raw_and_full_layers(training_item.input());
        let dot_sigma_z: Vec<Matrix<f64>> = (1..=layers)
            .map(|l| z[l].applying_to_all(&|x| neuralnet.activation_functions[l - 1].derivative(x)))
            .collect();

        let mut gradient_wrt_activations = a.clone(); // This is basically our DP table!

        // Base case of DP table, compute all dC/da for each activation in the final layer
        gradient_wrt_activations[layers] =
            (a[layers].clone() - training_item.correct_output()) * 2.0;
        gradient.derivatives.biases[layers - 1] =
            dot_sigma_z[layers - 1].hadamard(gradient_wrt_activations[layers].clone());
        gradient.derivatives.weights[layers - 1] =
            gradient.derivatives.biases[layers - 1].clone() * a[layers - 1].transpose();

        // the rest now! I want the indices to actually match the indices in the writeup as closely as possible.

        for layer in (0..layers).rev() {
            gradient_wrt_activations[layer] = neuralnet.weights[layer].transpose().clone()
                * dot_sigma_z[layer].hadamard(gradient_wrt_activations[layer + 1].clone());
            gradient.derivatives.biases[layer] =
                dot_sigma_z[layer].hadamard(gradient_wrt_activations[layer + 1].clone());
            gradient.derivatives.weights[layer] =
                gradient.derivatives.biases[layer].clone() * a[layer].transpose().clone();
        }

        gradient
    }

    /// Performs a step of GD on a mini-batch of data, returning the size
    /// of the gradient vector (before rescaling) so we can see how far from a local minimum we are.
    pub fn sgd_batch_step(
        &self,
        batch: Vec<DI>,
        neuralnet: &mut NeuralNet,
        learning_rate: f64,
    ) -> f64 {
        // First, compute sum of gradients for all training items in the batch.
        let mut gradient = NNGradient::from_nn_shape(neuralnet.clone());

        println!("Batch size: {}", batch.len());

        for item in batch {
            gradient += Self::compute_gradient(item, neuralnet);
        }

        let original_length = gradient.norm();

        // Now normalize that gradient!
        gradient.set_length(learning_rate);

        *neuralnet -= gradient;

        original_length
    }

    /// Runs Gradient Descent on this Data Set, outputting
    /// a neural network
    pub fn train_gd(
        &self,
        neuralnet: &mut NeuralNet,
        shape: Vec<usize>,
        learning_rate: f64,
        threshold: f64,
    ) {
    }

    /// Generates a random neural network of a particular shape
    pub fn random_network(&self, shape: Vec<usize>, activation_functions: Vec<AFI>) -> NeuralNet {
        let mut rand_gen = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap(); // Tweak as needed!

        let weights = (1..shape.len())
            .map(|layer| {
                Matrix::from_index_def(shape[layer], shape[layer - 1], &mut |_, _| {
                    normal.sample(&mut rand_gen)
                })
            })
            .collect();
        let biases = (1..shape.len())
            .map(|layer| {
                Matrix::from_index_def(shape[layer], 1, &mut |_, _| normal.sample(&mut rand_gen))
            })
            .collect();

        // Initialize a random neural network
        NeuralNet::new(weights, biases, activation_functions)
    }

    /// The average cost over all training examples
    pub fn cost(&self, network: &NeuralNet) -> f64 {
        let mut average_cost = 0.0;

        for item in self.data_set.data_items.clone() {
            let (x, y) = (item.input(), item.correct_output());
            let a = network.compute_final_layer(x);
            average_cost += (a - y).l2_norm_squared()
        }

        average_cost / (self.data_set.data_items.len() as f64)
    }
}

#[cfg(test)]
mod sgd_tests {
    use rand::Rng;
    use rand_distr::Uniform;

    use crate::{
        math::activation::AFI,
        utility::mnist::mnist_utility::{load_mnist, MNISTImage},
    };

    use super::SGDTrainer;

    #[test]
    fn test_network_stuff() {
        let dataset = load_mnist("train");
        let trainer = SGDTrainer::new(dataset);
        let mut network = trainer.random_network(
            vec![784, 16, 16, 10],
            vec![AFI::Sigmoid, AFI::Sigmoid, AFI::Sigmoid],
        );

        let learning_rate = 0.05;

        let original_cost = trainer.cost(&network);
        println!("Original NN cost: {}", original_cost);

        // Now, train it a bit!

        for i in 1..=1 {
            print!("Training iteration {}... ", i);

            trainer.sgd_batch_step(
                trainer.data_set.data_items[0..100].to_vec(),
                &mut network,
                learning_rate,
            );

            let new_cost = trainer.cost(&network);

            println!("cost is {}", new_cost);
        }
    }

    #[test]
    fn test_sgd() {
        println!("Starting test_sgd");
        let dataset: crate::training::dataset::DataSet<MNISTImage> = load_mnist("train");
        let trainer = SGDTrainer::new(dataset.clone());
        let mut network = trainer.random_network(
            vec![784, 16, 16, 10],
            vec![AFI::Sigmoid, AFI::Sigmoid, AFI::Sigmoid],
        );

        let learning_rate = 0.05;

        let original_cost = trainer.cost(&network);
        println!("Original NN cost: {}", original_cost);

        // Now, train it a bit!
        let epochs = 5;
        let rand_uni = rand_distr::Uniform::try_from(0..dataset.data_items.len()).unwrap();
        let mut rng = rand::rng();
        for i in 1..=epochs * dataset.data_items.len() {
            print!("Training iteration {}... ", i);
            let idx = rng.sample(rand_uni);
            print!("Random index {}", idx);
            trainer.sgd_batch_step(
                trainer.data_set.data_items[idx..=idx].to_vec(),
                &mut network,
                learning_rate,
            );

            let new_cost = trainer.cost(&network);

            println!("cost is {}", new_cost);
        }
    }
}
