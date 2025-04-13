use std::fs::File;

use ml_kit::{models::neuralnet::NeuralNet, training::dataset::{DataItem, DataSet}, utility::mnist::mnist_utility::load_mnist};



fn main() {
    
    let neural_net = match File::open("testing/files/digits_nn.mlk_nn") {
        Ok(mut f) => NeuralNet::from_file(&mut f),
        Err(e) => panic!("Error opening nn file: {:?}", e),
    };

    let dataset = load_mnist("t10k");

    for digit in dataset.random_sample(10) {
        println!("{:?}", digit);
        println!("Network says: {:?}", neural_net.classify(digit.input()));
    }

}