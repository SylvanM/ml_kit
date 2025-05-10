use core::panic;
use matrix_kit::dynamic::matrix::Matrix;
use rand_distr::Distribution;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Range, Sub, SubAssign,
};
use std::{
    fmt::Debug,
    fs::File,
    io::{Read, Write},
    process::exit,
    vec,
};

use crate::{math::activation::AFI, utility};
// use crate::{models::neuralnet::NeuralNet};

// pub type CNN = ConvNeuralNet;

pub enum Layer {
    Conv(ConvLayer),
    Pool(PoolLayer),
    Full(FullLayer),
}

pub struct ConvLayer {
    filters: Vec<Matrix<f64>>,
    biases: Matrix<f64>,
    act_func: AFI,
}

// For now assuming stride of 1 and thus trivial padding
impl ConvLayer {
    pub fn new(filters: Vec<Matrix<f64>>, biases: Matrix<f64>, act_func: AFI) -> ConvLayer {
        ConvLayer {
            filters,
            biases,
            act_func,
        }
    }

    /// Creates a new, empty convolutional layer with all filters and biases
    /// set to 0.
    pub fn zeros(
        filter_count: usize,
        filter_rows: usize,
        filter_cols: usize,
        act_func: AFI,
    ) -> ConvLayer {
        let filters = (1..filter_count)
            .map(|_| Matrix::new(filter_rows, filter_cols))
            .collect();
        let biases = Matrix::new(filter_count, 1);

        ConvLayer {
            filters,
            biases,
            act_func,
        }
    }

    /// Generates a random convolutional layer of a particular shape
    pub fn rand(
        filter_count: usize,
        filter_rows: usize,
        filter_cols: usize,
        act_func: AFI,
    ) -> ConvLayer {
        let mut rand_gen = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap(); // Tweak as needed!
        let mut conv_layer = ConvLayer::zeros(filter_count, filter_rows, filter_cols, act_func);

        for l in 0..filter_count {
            for r in 0..filter_rows {
                for c in 0..filter_cols {
                    conv_layer.filters[l].set(r, c, normal.sample(&mut rand_gen));
                }
            }

            conv_layer.biases.set(l, 0, normal.sample(&mut rand_gen));
        }

        conv_layer
    }

    /// Return the output of the layer given input matrix a
    /// (implementation following https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster)
    pub fn feedforward(&self, a: Matrix<f64>) -> Vec<Matrix<f64>> {
        let f_rows = self.filters[0].row_count();
        let f_cols = self.filters[0].col_count();

        // Convert convolution to matrix multiplication
        let mut cols: Vec<Matrix<f64>> = Vec::new();
        for r in 0..(a.row_count() - f_rows + 1) {
            for c in 0..(a.col_count() - f_cols + 1) {
                cols.push(Matrix::from_flatmap(
                    1,
                    f_rows * f_cols,
                    a.get_submatrix(
                        Range {
                            start: r,
                            end: r + f_rows,
                        },
                        Range {
                            start: c,
                            end: c + f_cols,
                        },
                    )
                    .as_vec(),
                ));
            }
        }
        let converted_input = Matrix::from_cols(cols);
        let converted_kernel = Matrix::from_cols(
            self.filters
                .clone()
                .into_iter()
                .map(|f| Matrix::from_flatmap(1, f_rows * f_cols, f.as_vec()))
                .collect(),
        )
        .transpose();
        let mut output: Vec<Matrix<f64>> = (converted_kernel * converted_input)
            .transpose()
            .columns()
            .into_iter()
            .map(|mat| {
                Matrix::from_flatmap(
                    a.row_count() - f_rows + 1,
                    a.col_count() - f_cols + 1,
                    mat.as_vec(),
                )
            })
            .collect();
        for l in 0..self.filters.len() {
            output[l].apply_to_all(&|x| self.act_func.evaluate(x + self.biases.get(l, 0)));
        }
        output
    }
}

pub struct PoolLayer {
    pool_cols: usize,
    pool_rows: usize,
}

pub struct FullLayer {
    weights: Matrix<f64>,
    biases: Matrix<f64>,
    act_func: AFI,
}

// #[derive(Clone)]
// pub struct ConvNeuralNet {
//     pub layers: Vec<Layer>,
// }
