//!
//! Stochastic Gradient Descent Trainer for Neural Networks
//!

use std::ops::{AddAssign, SubAssign};

use matrix_kit::dynamic::matrix::Matrix;

use crate::math::loss::LFI;
use crate::{math::activation::AFI, models::neuralnet::NeuralNet};

use super::dataset::{DataItem, DataSet};
use super::learning_rate::GradientUpdateSchedule;

use crate::models::convneuralnet::{ConvNeuralNet, Layer, CNNGradient};

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
#[derive(Clone, Debug)]
pub struct NNGradient {
    pub derivatives: NeuralNet,
}

/// A Gradient representation for CNNs
#[derive(Clone, Debug)]
pub struct CNNGradient {
    pub derivatives: Vec<Layer>,
}

impl SubAssign<NNGradient> for NeuralNet {
    fn sub_assign(&mut self, rhs: NNGradient) {
        for layer in 0..self.weights.len() {
            self.weights[layer] -= rhs.derivatives.weights[layer].clone();
            self.biases[layer] -= rhs.derivatives.biases[layer].clone();
        }
    }
}

impl SubAssign<CNNGradient> for ConvNeuralNet {
    fn sub_assign(&mut self, rhs: CNNGradient) {
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            match (layer, &rhs.derivatives[layer_idx]) {
                (Layer::Conv(conv), Layer::Conv(grad)) => {
                    // Conv Layers
                    for (filter_index, filter) in conv.filters.iter_mut().enumerate() {
                        for (depth_index, depth) in filter.iter_mut().enumerate() {
                            *depth -= grad.filters[filter_index][depth_index].clone();
                        }
                    }
                    conv.biases -= grad.biases.clone();
                }
                (Layer::Full(full), Layer::Full(grad)) => {
                    // FC Layers
                    full.weights -= grad.weights.clone();
                    full.biases -= grad.biases.clone();
                }
                _ => {} // Pool layers
            }
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

impl AddAssign for CNNGradient {
    fn add_assign(&mut self, rhs: Self) {
        for (layer_index, layer) in self.derivatives.iter_mut().enumerate() {
            match (layer, &rhs.derivatives[layer_index]) {
                (Layer::Conv(conv), Layer::Conv(grad)) => {
                    // Conv Layers
                    for (filter_index, filter) in conv.filters.iter_mut().enumerate() {
                        for (depth_index, depth) in filter.iter_mut().enumerate() {
                            *depth += grad.filters[filter_index][depth_index].clone();
                        }
                    }
                    conv.biases += grad.biases.clone();
                }
                (Layer::Full(full), Layer::Full(grad)) => {
                    // FClayers
                    full.weights += grad.weights.clone();
                    full.biases += grad.biases.clone();
                }
                _ => {} // Pool layers
            }
        }
    }
}

impl PartialEq for NNGradient {
    fn eq(&self, other: &Self) -> bool {
        if self.derivatives.shape() != other.derivatives.shape() {
            return false;
        } else {
            for l in 0..self.derivatives.weights.len() {
                if self.derivatives.weights[l] != other.derivatives.weights[l] {
                    return false;
                }

                if self.derivatives.biases[l] != other.derivatives.biases[l] {
                    return false;
                }
            }

            return true;
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

    pub fn as_vec(&self) -> Matrix<f64> {
        let mut grad = Matrix::new(self.derivatives.parameter_count(), 1);

        let mut i = 0;
        for l in 0..self.derivatives.weights.len() {
            for r in 0..self.derivatives.weights[l].row_count() {
                for c in 0..self.derivatives.weights[l].col_count() {
                    grad.set(i, 0, self.derivatives.weights[l].get(r, c));
                    i += 1;
                }
            }

            for b in 0..self.derivatives.biases[l].row_count() {
                grad.set(i, 0, self.derivatives.biases[l].get(b, 0));
                i += 1;
            }
        }

        debug_assert_eq!(i, self.derivatives.parameter_count());

        grad
    }

    pub fn from_vec(grad: Matrix<f64>, shape: Vec<usize>) -> NNGradient {
        let mut derivatives =
            NeuralNet::from_shape(shape.clone(), vec![AFI::Identity; shape.len() - 1]);

        let mut i = 0;
        for l in 0..derivatives.weights.len() {
            for r in 0..derivatives.weights[l].row_count() {
                for c in 0..derivatives.weights[l].col_count() {
                    derivatives.weights[l].set(r, c, grad.get(i, 0));
                    i += 1;
                }
            }

            for b in 0..derivatives.biases[l].row_count() {
                derivatives.biases[l].set(b, 0, grad.get(i, 0));
                i += 1;
            }
        }

        NNGradient { derivatives }
    }
}

impl CNNGradient {
    pub fn from_cnn_shape(cnn: &ConvNeuralNet) -> CNNGradient {
        let mut derivatives = Vec::new();
        
        for layer in &cnn.layers {
            match layer {
                Layer::Conv(conv) => {
                    // Conv layer -> zero gradients
                    let mut zero_filters = Vec::new();
                    for filter in &conv.filters {
                        let mut zero_filter = Vec::new();
                        for depth in filter {
                            zero_filter.push(Matrix::new(depth.row_count(), depth.col_count()));
                        }
                        zero_filters.push(zero_filter);
                    }
                    derivatives.push(Layer::Conv(ConvLayer::new(
                        zero_filters,
                        Matrix::new(conv.biases.row_count(), 1),
                        conv.act_func.clone(),
                        conv.stride,
                        conv.padding,
                    )));
                }
                Layer::Pool(pool) => {
                    // Pool layers -> no params 
                    derivatives.push(Layer::Pool(PoolLayer::new(
                        pool.pool_type.clone(),
                        pool.w_rows,
                        pool.w_cols,
                        pool.stride,
                    )));
                }
                Layer::Full(full) => {
                    // Fc Layer -> zero gradients
                    derivatives.push(Layer::Full(FullLayer::new(
                        Matrix::new(full.weights.row_count(), full.weights.col_count()),
                        Matrix::new(full.biases.row_count(), 1),
                        full.act_func.clone(),
                    )));
                }
            }
        }
        
        CNNGradient { derivatives }
    }

    pub fn norm(&self) -> f64 {
        let mut norm_squared = 0.0;
        
        for layer in &self.derivatives {
            match layer {
                Layer::Conv(conv) => {
                    // sum^2 norms 
                    for filter in &conv.filters {
                        for depth in filter {
                            norm_squared += depth.l2_norm_squared();
                        }
                    }
                    norm_squared += conv.biases.l2_norm_squared();
                }
                Layer::Full(full) => {
                    norm_squared += full.weights.l2_norm_squared();
                    norm_squared += full.biases.l2_norm_squared();
                }
                _ => {} 
            }
        }
        
        norm_squared.sqrt()
    }

    pub fn set_length(&mut self, length: f64) {
        let norm = self.norm();
        if norm == 0.0 {
            return;
        }
        
        for layer in &mut self.derivatives {
            match layer {
                // Conv Layer
                Layer::Conv(conv) => {
                    for filter in &mut conv.filters {
                        for depth in filter {
                            *depth = depth.clone() * (length / norm);
                        }
                    }
                    conv.biases = conv.biases.clone() * (length / norm);
                }
                // FCLayer
                Layer::Full(full) => {
                    full.weights = full.weights.clone() * (length / norm);
                    full.biases = full.biases.clone() * (length / norm);
                }
                _ => {} //Pool Layer
            }
        }
    }
}

#[cfg(test)]
mod grad_tests {
    use rand::Rng;

    use crate::{math::activation::AFI, models::neuralnet::NeuralNet};

    use super::NNGradient;

    #[test]
    fn test_creation_inversion() {
        let mut rng = rand::rng();

        for _ in 0..10 {
            let layers = rng.random_range(3..100);
            let mut shape = vec![0; layers];
            for l in 0..layers {
                shape[l] = rng.random_range(3..100);
            }

            let derivatives =
                NeuralNet::random_network(shape.clone(), vec![AFI::Identity; layers - 1]);

            let gradient = NNGradient { derivatives };
            let grad_vector = gradient.as_vec();
            let new_gradient = NNGradient::from_vec(grad_vector, shape);

            assert_eq!(gradient, new_gradient);
        }
    }
}

impl<DI: DataItem> SGDTrainer<DI> {
    pub fn new(
        training_data_set: DataSet<DI>,
        testing_data_set: DataSet<DI>,
        loss_function: LFI,
    ) -> SGDTrainer<DI> {
        SGDTrainer {
            training_data_set,
            testing_data_set,
            loss_function,
        }
    }

    // MARK: Training

    /// Computes the gradient for a single training sample. The "gradient" is
    /// represented as its own neural network, but a "derivative" neural network,
    /// if you will. (This just keeps everything organized)
    ///
    /// TODO: Generalize this to think about more than just squared loss
    pub fn compute_gradient(&self, training_item: DI, neuralnet: &NeuralNet) -> NNGradient {
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
        gradient_wrt_activations[layers] = self
            .loss_function
            .derivative(&a[layers], &training_item.correct_output());
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
    pub fn sgd_batch_step<GUS: GradientUpdateSchedule>(
        &self,
        batch: Vec<DI>,
        neuralnet: &mut NeuralNet,
        gus: &mut GUS,
    ) -> f64 {
        // First, compute sum of gradients for all training items in the batch.
        let mut gradient = NNGradient::from_nn_shape(neuralnet.clone());

        for item in batch {
            gradient += self.compute_gradient(item, neuralnet);
        }

        let original_length = gradient.norm();

        // Now normalize that gradient!
        gus.next_gradient(&mut gradient);

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
    pub fn train_sgd<GUS: GradientUpdateSchedule>(
        &self,
        neuralnet: &mut NeuralNet,
        gus: &mut GUS,
        epochs: usize,
        batch_size: usize,
        verbose: bool,
    ) {
        // Repeat for all epochs!

        for epoch in 0..epochs {
            if verbose {
                println!("Training on epoch {}...", epoch);
            }

            for batch in self.training_data_set.all_minibatches(batch_size) {
                self.sgd_batch_step(batch, neuralnet, gus);
            }
        }

        if verbose {
            println!("Completed all epochs of training.");
        }
    }

    /// TODO: integrate w/ backprop.
    pub fn compute_cnn_gradient(&self, training_item: DI, cnn: &ConvNeuralNet) -> CNNGradient {
    }
             

    /// Performs a step of SGD on a mini-batch of data for a CNN
    pub fn sgd_cnn_batch_step<GUS: GradientUpdateSchedule>(
        &self,
        batch: Vec<DI>,
        cnn: &mut ConvNeuralNet,
        gus: &mut GUS,
    ) -> f64 {
        let mut gradient = CNNGradient::from_cnn_shape(cnn);

        for item in batch {
            gradient += self.compute_cnn_gradient(item, cnn);
        }

        let original_length = gradient.norm();

        gus.next_gradient(&mut gradient);

        *cnn -= gradient;

        original_length
    }

    /// Trains a CNN using SGD
    pub fn train_cnn_sgd<GUS: GradientUpdateSchedule>(
        &self,
        cnn: &mut ConvNeuralNet,
        gus: &mut GUS,
        epochs: usize,
        batch_size: usize,
        verbose: bool,
    ) {
        for epoch in 0..epochs {
            if verbose {
                println!("Training CNN on epoch {}...", epoch);
            }

            for batch in self.training_data_set.all_minibatches(batch_size) {
                self.sgd_cnn_batch_step(batch, cnn, gus);
            }
        }

        if verbose {
            println!("Completed all epochs of CNN training.");
        }
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
        println!(
            "Displaying network performance on {} testing items",
            num_items
        );

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
    fn test_stuff() {}
}