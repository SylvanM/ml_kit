use ml_kit::utility::file_utility::{load_mnist, MnistImage};

fn print_image(image: &MnistImage) {
    println!("");
    for r in 0..image.image.row_count() {
        for c in 0..image.image.col_count() {
            if image.image.get(c, r) <= 0.2 {
                print!(" ")
            } else if image.image.get(c, r) <= 0.4 {
                print!("░")
            } else if image.image.get(c, r) <= 0.6 {
                print!("▒")
            } else if image.image.get(c, r) <= 0.8 {
                print!("▓")
            } else {
                print!("█")
            }
        }
        println!("");
    }
    println!("{}", image.label);
}

fn main() {
    let data = load_mnist("train");
    // note: image matrix is transposed (hard to find issue: could be parser, matrix library, or printer)
    for i in 100..110 {
        print_image(&data[i]);
    }
}
