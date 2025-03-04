use idx_parser::IDXFile;
use matrix_kit::dynamic::matrix::Matrix;
use std::{fs::File, io::prelude::*};

/// Converts floats to bytes, using Big Endian format
pub fn floats_to_bytes(vec: Vec<f64>) -> Vec<u8> {
    vec.iter().map(|x| x.to_be_bytes()).flatten().collect()
}

/// Converts a vector of bytes to floats, using Big Endian
pub fn bytes_to_floats(vec: Vec<u8>) -> Vec<f64> {
    let mut floats = vec![0f64; vec.len() / 8];

    for i in 0..floats.len() {
        floats[i] = f64::from_be_bytes(vec[(i * 8)..((i + 1) * 8)].try_into().unwrap())
    }

    floats
}

/// Converts u64s to bytes, using Big Endian format
pub fn u64s_to_bytes(vec: Vec<u64>) -> Vec<u8> {
    vec.iter().map(|x| x.to_be_bytes()).flatten().collect()
}

/// Converts a vector of bytes to u64s, using Big Endian
pub fn bytes_to_u64s(vec: Vec<u8>) -> Vec<u64> {
    let mut ints = vec![0u64; vec.len() / 8];

    for i in 0..ints.len() {
        ints[i] = u64::from_be_bytes(vec[(i * 8)..((i + 1) * 8)].try_into().unwrap())
    }

    ints
}

pub struct MnistImage {
    pub image: Matrix<f64>,
    pub label: u8,
}

pub fn load_mnist(dataset_name: &str) -> Vec<MnistImage> {
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

    let mut data: Vec<MnistImage> = vec![];

    // loop through each image
    for i in 0..images_idx.matrix_data.len() {
        // set label
        let label = (*labels_idx.matrix_data[i])
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

                flattened.push(val as f64); // add entry to flattened matrix
            }
        }

        let image = Matrix::from_flatmap(n_rows, n_cols, flattened); // create matrix of image data
        data.push(MnistImage {
            label: label,
            image: image,
        }); // add label and image to vector of output data
    }

    data
}

#[cfg(test)]
mod file_tests {
    use crate::utility::file_utility::{
        bytes_to_floats, bytes_to_u64s, floats_to_bytes, u64s_to_bytes,
    };

    #[test]
    fn test_byte_conversion() {
        let float_vec = vec![0.3453, 0.3467245372, 123513.1462456257752];
        debug_assert_eq!(
            float_vec,
            bytes_to_floats(floats_to_bytes(float_vec.clone()))
        );

        let int_vec = vec![23532, 6246, 0000, 0465345];
        debug_assert_eq!(int_vec, bytes_to_u64s(u64s_to_bytes(int_vec.clone())));
    }
}
