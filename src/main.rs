use ml_kit::utility::mnist::mnist_utility::*;

fn main() {
    let data = load_mnist("train");
    // note: image matrix is transposed (hard to find issue: could be parser, matrix library, or printer)
    for i in 100..110 {
        println!("{:?}", data.data_items[i])
    }
}