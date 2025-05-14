use std::fs::File;

use ml_kit::math::activation::AFI;
use ml_kit::models::neuralnet::NeuralNet;
//use ml_kit::training::learning_rate::FixedLR;
use ml_kit::training::learning_rate::AdaGrad;
use ml_kit::{
    math::loss::LFI, training::sgd::SGDTrainer, utility::mnist::mnist_utility::load_mnist,
};

fn main() {
    let relative_path: &'static str = "../data_sets";

    let dataset = load_mnist(relative_path, "train");
    let testing_ds = load_mnist(relative_path, "t10k");
    let trainer: SGDTrainer<ml_kit::utility::mnist::mnist_utility::MNISTImage> =
        SGDTrainer::new(dataset, testing_ds, LFI::Squared);

    let mut neuralnet = NeuralNet::random_network(
        vec![784, 16, 16, 10],
        vec![AFI::Sigmoid, AFI::Sigmoid, AFI::Sigmoid],
    );

    //let mut grad_update_sched = FixedLR::new(0.05);
    let mut grad_update_sched = AdaGrad::new(0.01, neuralnet.parameter_count());
    let epochs = 100;

    let original_cost = trainer.cost(&neuralnet);
    println!("Original cost: {}", original_cost);

    trainer.train_sgd(&mut neuralnet, &mut grad_update_sched, epochs, 32, true);

    let final_cost = trainer.cost(&neuralnet);

    println!("Final cost: {}", final_cost);

    // Now, let's go through and actually try it out!

    trainer.display_behavior(&neuralnet, 10);

    println!("Writing final network to testing folder.");

    match File::create("testing/files/digits.mlk_nn") {
        Ok(mut f) => neuralnet.write_to_file(&mut f),
        Err(e) => println!("Error writing to file: {:?}", e),
    }
}
