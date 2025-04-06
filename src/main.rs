use ml_kit::{math::activation::AFI, training::{self, sgd::SGDTrainer}, utility::mnist::mnist_utility::{load_mnist, MNISTImage}};



fn main() {
    let dataset = load_mnist("train");
    let trainer = SGDTrainer::new(dataset);

    trainer.train_sgd(vec![784, 16, 16, 10], vec![AFI::Sigmoid, AFI::Sigmoid, AFI::Sigmoid]);
}