# ML Kit

`ml_kit` is an open-source Machine Learning library for Rust! 

## Quickstart 

Go ahead and download the MNIST Digits database, put it in a folder in your project
so that the image and label files can be accessed via the path `data/digits/FILENAME.idx{1,3}-ubyte`.

After having added `ml_kit` to your project via something like `cargo add ml_kit`,
you can run the following code to quickly train a Neural Network on 
images of handwritten digits.

```rust
use std::fs::File;

use ml_kit::{math::LFI, training::sgd::SGDTrainer, utility::mnist::mnist_utility::load_mnist};
use ml_kit::math::activation::AFI;

fn main() {

    let relative_path = "../Data sets/MNIST/digits";

    let dataset = load_mnist(relative_path, "train");
    let testing_ds = load_mnist(relative_path, "t10k");
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

    match File::create("testing/files/digits.mlk_nn") {
        Ok(mut f) => neuralnet.write_to_file(&mut f),
        Err(e) => println!("Error writing to file: {:?}", e),
    }

}
```

In the end, the behavior of the network will be printed to the screen, and a 
file representing the parameters of the network is written to
`testing/files/digits.mlk_nn`.