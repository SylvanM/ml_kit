use std::{fs::File, vec};

use ml_kit::{math::{activation::AFI, LFI}, models::neuralnet, training::{self, sgd::SGDTrainer}, utility::mnist::mnist_utility::{load_mnist, MNISTImage}};



fn main() {
    let dataset = load_mnist("train");
    let testing_ds = load_mnist("t10k");
    let trainer = SGDTrainer::new(dataset, testing_ds, LFI::Squared);

    let mut neuralnet = trainer.random_network(vec![784, 16, 16, 10], vec![AFI::Sigmoid, AFI::Sigmoid, AFI::Sigmoid]);
 
    let learning_rate = 0.05;
    let epochs = 100;

    let original_cost = trainer.cost(&neuralnet);
    println!("Original cost: {}", original_cost);

    trainer.train_sgd(&mut neuralnet, learning_rate, epochs, 32);

    let final_cost = trainer.cost(&neuralnet);

    println!("Final cost: {}", final_cost);

    // Now, let's go through and actually try it out!

    trainer.display_behavior(&neuralnet, 10);

    println!("Writing final network to testing folder.");

    match File::create("testing/files/digits_nn.mlk_nn") {
        Ok(mut f) => neuralnet.write_to_file(&mut f),
        Err(e) => println!("Error writing to file: {:?}", e),
    }


}