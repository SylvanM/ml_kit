use ml_kit::{
    math::activation::AFI,
    math::loss::SquaredLoss,
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
    let loss_fn = SquaredLoss {};
    let epochs: usize = 5;
    let rand_uni: Uniform<usize> = rand_distr::Uniform::try_from(0..dataset.data_items.len()).unwrap();
    let mut rng: rand::prelude::ThreadRng = rand::rng();
    for i in 1..=epochs * dataset.data_items.len() {
        let idx = rand_uni.sample(&mut rng);

        trainer.sgd_batch_step(
            trainer.data_set.data_items[idx..=idx].to_vec(),
            &mut network,
            learning_rate,
            &loss_fn,
        );

        if i % 10000 == 0 {
            println!("Iteration: {}", i);
        }

        if i % 60000 == 0 {
            println!("Epoch {} cost: {}", i / 60000, trainer.cost(&network));
        }
    }

    println!("Final NN cost: {}", trainer.cost(&network));
}
