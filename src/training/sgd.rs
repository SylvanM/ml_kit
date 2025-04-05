//!
//! Stochastic Gradient Descent Trainer for Neural Networks
//!

use crate::models::neuralnet::NeuralNet;


/// An SGD trainer that trains a nerual network
pub struct SGDTrainer<'a> {
    
    /// A reference to the neural network being trained
    pub neural_net: &'a NeuralNet,

    /// A dataset on which to train

}

impl<'a>  SGDTrainer<'a>  {

    pub fn new(neural_net: &NeuralNet) -> SGDTrainer {
        SGDTrainer { neural_net }
    }

}