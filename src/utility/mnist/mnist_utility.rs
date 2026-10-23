use matrix_kit::dynamic::matrix::Matrix;
use std::{fmt::Debug, fs::File, io::prelude::*};

use crate::training::dataset::{DataItem, DataSet};

#[derive(Clone)]
pub struct MNISTImage {
    /// A 28x28 grayscale image of a handwritten digit. The [0,0] entry is
    /// the top-left corner of the image.
    image_matrix: Matrix<f64>,

    /// The digit this image represents
    correct_digit: usize,
}

impl Debug for MNISTImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n")?;
        for r in 0..self.image_matrix.row_count() {
            for c in 0..self.image_matrix.col_count() {
                if self.image_matrix.get(c, r) <= 0.2 {
                    write!(f, " ")?;
                } else if self.image_matrix.get(c, r) <= 0.4 {
                    write!(f, "░")?;
                } else if self.image_matrix.get(c, r) <= 0.6 {
                    write!(f, "▒")?;
                } else if self.image_matrix.get(c, r) <= 0.8 {
                    write!(f, "▓")?;
                } else {
                    write!(f, "█")?;
                }
            }
            write!(f, "\n")?;
        }
        write!(f, "\n{}", self.correct_digit)
    }
}

impl DataItem for MNISTImage {
    fn input(&self) -> Matrix<f64> {
        // We store the image in a flat vector, column-wise. So, the vector
        // is 784-dimensional, with the first 28 entries being the first
        // column of the image.

        Matrix::from_flatmap(784, 1, self.image_matrix.as_vec())
    }

    fn correct_output(&self) -> Matrix<f64> {
        // Returns a 1-hot encoding of the correct digit. This is a vector
        // with 10 entries, with index i being 1 iff this represents the image
        // of the digit i.

        let mut one_hot: Matrix<f64> = Matrix::new(10, 1);
        one_hot.set(self.correct_digit, 0, 1.0);

        one_hot
    }

    fn name(&self) -> String {
        self.correct_digit.to_string()
    }

    fn label(&self) -> usize {
        self.correct_digit
    }
}

pub fn load_mnist(relative_path: &str, prefix: &str) -> DataSet<MNISTImage> {
    // get image data
    let (dimensions, pixels) =
        read_idx_bytes(&format!("{}/{}-images.idx3-ubyte", relative_path, prefix));
    assert_eq!(dimensions.len(), 3, "MNIST parsing error: images file isn't 3-dimensional");
    let (n_images, n_rows, n_cols) = (dimensions[0], dimensions[1], dimensions[2]);

    // get label data
    let (_, labels) = read_idx_bytes(&format!("{}/{}-labels.idx1-ubyte", relative_path, prefix));
    assert_eq!(labels.len(), n_images, "MNIST parsing error: number of labels and images differ");

    let mut data: Vec<MNISTImage> = vec![];

    // loop through each image
    for i in 0..n_images {
        // the image's pixels, one row after another
        let image_bytes = &pixels[(i * n_rows * n_cols)..((i + 1) * n_rows * n_cols)];
        let flattened: Vec<f64> = image_bytes.iter().map(|&val| (val as f64) / 255.0).collect();

        let image = Matrix::from_flatmap(n_rows, n_cols, flattened); // create matrix of image data
        data.push(MNISTImage {
            correct_digit: labels[i] as usize,
            image_matrix: image,
        }); // add label and image to vector of output data
    }

    DataSet { data_items: data }
}

/// Reads an IDX file of unsigned bytes (the format of the MNIST files),
/// returning the size of each dimension along with the data.
///
/// The file starts with two zero bytes, a byte for the data type (0x08 means
/// unsigned bytes), and a byte for the number of dimensions. Then comes each
/// dimension's size as a big-endian u32, then the data in row-major order.
fn read_idx_bytes(path: &str) -> (Vec<usize>, Vec<u8>) {
    let mut bytes: Vec<u8> = vec![];
    let mut file = File::open(path).expect(&format!("Unable to open MNIST file at path {:?}", path));
    file.read_to_end(&mut bytes).unwrap();

    assert!(
        bytes.len() >= 4 && bytes[0] == 0 && bytes[1] == 0,
        "MNIST parsing error: {:?} isn't an IDX file", path
    );
    assert_eq!(bytes[2], 0x08, "MNIST parsing error: {:?} doesn't hold unsigned bytes", path);

    let data_start = 4 + 4 * bytes[3] as usize;
    assert!(bytes.len() >= data_start, "MNIST parsing error: {:?} is cut off", path);

    let dimensions: Vec<usize> = bytes[4..data_start]
        .chunks_exact(4)
        .map(|size| u32::from_be_bytes([size[0], size[1], size[2], size[3]]) as usize)
        .collect();

    let data_size: usize = dimensions.iter().product();
    assert!(bytes.len() >= data_start + data_size, "MNIST parsing error: {:?} is cut off", path);

    (dimensions, bytes[data_start..(data_start + data_size)].to_vec())
}

#[cfg(test)]
mod mnust_utility_tests {
    use super::load_mnist;

    #[test]
    fn test_printing() {
        let data = load_mnist("fashion", "train");
        // note: image matrix is transposed (hard to find issue: could be parser, matrix library, or printer)
        for i in 100..110 {
            println!("{:?}", data.data_items[i])
        }
    }
}
