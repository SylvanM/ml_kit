use ml_kit::{
    math::activation::AFI,
    training::{self, sgd::SGDTrainer},
    utility::mnist::mnist_utility::{load_mnist, MNISTImage},
};
use rand_distr::Distribution;
use rand_distr::Uniform;

fn main() {
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
    let mut rng: rand::prelude::ThreadRng = rand::rng();
    for i in 1..=epochs * dataset.data_items.len() {
        let idx = rand_uni.sample(&mut rng);

        trainer.sgd_batch_step(
            trainer.data_set.data_items[idx..=idx].to_vec(),
            &mut network,
            learning_rate,
        );

        if i % dataset.data_items.len() == 0 {
            println!("Iteration: {}", i);
        }
    }

    println!("Final NN cost: {}", trainer.cost(&network));
}
