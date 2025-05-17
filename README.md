# ML Kit

`ml_kit` is an open-source Machine Learning library for Rust! 

## Quickstart 

Go ahead and download the MNIST Digits database, put it in a folder in your project
so that the image and label files can be accessed via the path `data/digits/FILENAME.idx{1,3}-ubyte`.

After having added `ml_kit` to your project via something like `cargo add ml_kit`,
you can run the following code to quickly train a Neural Network on 
images of handwritten digits.

```rust
use ml_kit::{math::{activation::AFI, loss::LFI}, models::neuralnet::NeuralNet, training::{learning_rate::FixedLR, sgd::SGDTrainer}, utility::mnist::mnist_utility::load_mnist};

fn main() {
    let path = "../Data Sets/MNIST/digits";

    let dataset = load_mnist(path, "train");
    let testing_dataset = load_mnist(path, "t10k");

    let trainer = SGDTrainer::new(dataset, testing_dataset, LFI::Squared);

    let mut neuralnet = NeuralNet::random_network(
        vec![784, 16, 16, 10], 
        vec![AFI::Sigmoid ; 3],
    );

    let mut gradient_update = FixedLR::new(0.05);

    let epochs = 100;

    let original_cost = trainer.cost(&neuralnet);

    println!("Original cost: {}", original_cost);

    trainer.train_sgd(&mut neuralnet, &mut gradient_update, epochs, 32, true);

    let final_cost = trainer.cost(&neuralnet);

    println!("Final cost: {}", final_cost);

    trainer.display_behavior(&neuralnet, 10);
}
```

In the end, the behavior of the network will be printed to the screen!
