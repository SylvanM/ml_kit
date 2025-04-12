use ml_kit::{math::{activation::AFI, LFI}, training::{self, sgd::SGDTrainer}, utility::mnist::mnist_utility::{load_mnist, MNISTImage}};



fn main() {
    let dataset = load_mnist("train");
    let trainer = SGDTrainer::new(dataset, LFI::Squared);
}