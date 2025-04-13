use idx_parser::IDXFile;
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

        let mut one_hot = Matrix::new(10, 1);
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

pub fn load_mnist(dataset_name: &str) -> DataSet<MNISTImage> {
    // get image data
    let mut buf: Vec<u8> = vec![];
    let mut file = File::open(format!("data/{}-images.idx3-ubyte", dataset_name)).unwrap();
    file.read_to_end(&mut buf).unwrap();
    let images_idx: IDXFile = IDXFile::from_bytes(buf).unwrap();
    let n_rows: usize = images_idx.dimensions[1].try_into().unwrap();
    let n_cols: usize = images_idx.dimensions[2].try_into().unwrap();

    // get label data
    let mut buf: Vec<u8> = vec![];
    let mut file = File::open(format!("data/{}-labels.idx1-ubyte", dataset_name)).unwrap();
    file.read_to_end(&mut buf).unwrap();
    let labels_idx = IDXFile::from_bytes(buf).unwrap();

    let mut data: Vec<MNISTImage> = vec![];

    // loop through each image
    for i in 0..images_idx.matrix_data.len() {
        // set label
        let label: u8 = (*labels_idx.matrix_data[i])
            .clone()
            .try_into()
            .expect("MNIST parsing error");

        // initialize flattened matrix of image data
        let mut flattened: Vec<f64> = vec![];

        // vector of rows of matrix
        let rows: Vec<Box<idx_parser::matrix::Matrix>> = (*images_idx.matrix_data[i])
            .clone()
            .try_into()
            .expect("MNIST parsing error");

        // loop through each row of matrix
        for row in rows {
            // vector of entries of current row of matrix
            let inner_rows: Vec<Box<idx_parser::matrix::Matrix>> =
                (*row).clone().try_into().expect("MNIST parsing error");

            // loop through each entry of current row of matrix
            for inner_row in inner_rows {
                let val: u8 = (*inner_row)
                    .clone()
                    .try_into()
                    .expect("MNIST parsing error"); // get entry value

                flattened.push((val as f64) / 255.0); // add entry to flattened matrix
            }
        }

        let image = Matrix::from_flatmap(n_rows, n_cols, flattened); // create matrix of image data
        data.push(MNISTImage {
            correct_digit: label as usize,
            image_matrix: image,
        }); // add label and image to vector of output data
    }

    DataSet { data_items: data }
}

#[cfg(test)]
mod mnust_utility_tests {
    use super::load_mnist;

    #[test]
    fn test_printing() {
        let data = load_mnist("train");
        // note: image matrix is transposed (hard to find issue: could be parser, matrix library, or printer)
        for i in 100..110 {
            println!("{:?}", data.data_items[i])
        }
    }
}