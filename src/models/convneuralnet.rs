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

#[derive(Clone)]
pub enum Layer {
    Conv(ConvLayer),
    Pool(PoolLayer),
    Full(FullLayer),
}

#[derive(Clone)]
pub struct ConvLayer {
    filters: Vec<Matrix<f64>>,
    biases: Matrix<f64>,
    act_func: AFI,
    stride: usize, 
    padding: usize, 
}

impl ConvLayer {
    pub fn new(filters: Vec<Matrix<f64>>, biases: Matrix<f64>, act_func: AFI, stride: usize, padding: usize) -> ConvLayer {
        ConvLayer {
            filters,
            biases,
            act_func,
            stride,
            padding,
        }
    }

    /// Creates a new, empty convolutional layer with all filters and biases
    /// set to 0.
    pub fn zeros(
        filter_count: usize,
        filter_rows: usize,
        filter_cols: usize,
        act_func: AFI,
        stride: usize,
        padding: usize,
    ) -> ConvLayer {
        let filters = (0..filter_count)
            .map(|_| Matrix::new(filter_rows, filter_cols))
            .collect();
        let biases = Matrix::new(filter_count, 1);

        ConvLayer {
            filters,
            biases,
            act_func,
            stride,
            padding,
        }
    }

    /// Generates a random convolutional layer of a particular shape
    pub fn rand(
        filter_count: usize,
        filter_rows: usize,
        filter_cols: usize,
        act_func: AFI,
        stride: usize,
        padding: usize,
    ) -> ConvLayer {
        let mut rand_gen = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap(); // Tweak as needed!
        let mut conv_layer = ConvLayer::zeros(filter_count, filter_rows, filter_cols, act_func, stride, padding);

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

    /// Adds padding to the input matrix
    fn add_padding(&self, input: &Matrix<f64>) -> Matrix<f64> {
        if self.padding == 0 {
            return input.clone();
        }

        let padded_rows = input.row_count() + 2 * self.padding;
        let padded_cols = input.col_count() + 2 * self.padding;
        let mut padded = Matrix::new(padded_rows, padded_cols);

        // Copy original input to the center of padded matrix
        for i in 0..input.row_count() {
            for j in 0..input.col_count() {
                padded.set(i + self.padding, j + self.padding, input.get(i, j));
            }
        }
        padded
    }

    /// Return the output of the layer given input matrix a
    pub fn feedforward(&self, a: &Matrix<f64>) -> Vec<Matrix<f64>> {
        let f_rows = self.filters[0].row_count();
        let f_cols = self.filters[0].col_count();

        // Add padding if needed
        let padded_input = self.add_padding(a);
        
        // Calculate output dimensions
        let output_rows = (padded_input.row_count() - f_rows) / self.stride + 1;
        let output_cols = (padded_input.col_count() - f_cols) / self.stride + 1;

        // Convert convolution to matrix multiplication
        let mut cols: Vec<Matrix<f64>> = Vec::new();
        for r in (0..output_rows).step_by(self.stride) {
            for c in (0..output_cols).step_by(self.stride) {
                cols.push(Matrix::from_flatmap(
                    1,
                    f_rows * f_cols,
                    padded_input.get_submatrix(
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
                    output_rows,
                    output_cols,
                    mat.as_vec(),
                )
            })
            .collect();

        // Apply activation and add bias
        for l in 0..self.filters.len() {
            output[l].apply_to_all(&|x| self.act_func.evaluate(x + self.biases.get(l, 0)));
        }
        output
    }
}

#[derive(Clone)]
pub enum PoolType {
    Max,
    Average,
}

#[derive(Clone)]
pub struct PoolLayer {
    pool_type: PoolType,
    window_size: usize,
    stride: usize,
}

impl PoolLayer {
    pub fn new(pool_type: PoolType, window_size: usize, stride: usize) -> PoolLayer {
        PoolLayer {
            pool_type,
            window_size,
            stride,
        }
    }

    pub fn forward(&self, input: &Matrix<f64>) -> Matrix<f64> {
        match self.pool_type {
            PoolType::Max => self.max_pool(input),
            PoolType::Average => self.avg_pool(input),
        }
    }

    fn max_pool(&self, input: &Matrix<f64>) -> Matrix<f64> {
        let output_rows = (input.row_count() - self.window_size) / self.stride + 1;
        let output_cols = (input.col_count() - self.window_size) / self.stride + 1;
        let mut result = Matrix::new(output_rows, output_cols);
        
        for i in 0..output_rows {
            for j in 0..output_cols {
                let mut max_val = f64::NEG_INFINITY;
                for wi in 0..self.window_size {
                    for wj in 0..self.window_size {
                        let input_i = i * self.stride + wi;
                        let input_j = j * self.stride + wj;
                        if input_i < input.row_count() && input_j < input.col_count() {
                            max_val = max_val.max(input.get(input_i, input_j));
                        }
                    }
                }
                result.set(i, j, max_val);
            }
        }
        result
    }

    fn avg_pool(&self, input: &Matrix<f64>) -> Matrix<f64> {
        let output_rows = (input.row_count() - self.window_size) / self.stride + 1;
        let output_cols = (input.col_count() - self.window_size) / self.stride + 1;
        let mut result = Matrix::new(output_rows, output_cols);
        
        for i in 0..output_rows {
            for j in 0..output_cols {
                let mut sum = 0.0;
                let mut count = 0;
                for wi in 0..self.window_size {
                    for wj in 0..self.window_size {
                        let input_i = i * self.stride + wi;
                        let input_j = j * self.stride + wj;
                        if input_i < input.row_count() && input_j < input.col_count() {
                            sum += input.get(input_i, input_j);
                            count += 1;
                        }
                    }
                }
                result.set(i, j, sum / count as f64);
            }
        }
        result
    }
}

#[derive(Clone)]
pub struct FullLayer {
    weights: Matrix<f64>,
    biases: Matrix<f64>,
    act_func: AFI,
}

impl FullLayer {
    pub fn new(weights: Matrix<f64>, biases: Matrix<f64>, act_func: AFI) -> FullLayer {
        FullLayer {
            weights,
            biases,
            act_func,
        }
    }

    pub fn zeros(input_size: usize, output_size: usize, act_func: AFI) -> FullLayer {
        let weights = Matrix::new(output_size, input_size);
        let biases = Matrix::new(output_size, 1);
        FullLayer::new(weights, biases, act_func)
    }

    pub fn rand(input_size: usize, output_size: usize, act_func: AFI) -> FullLayer {
        let mut rand_gen = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        
        let mut weights = Matrix::new(output_size, input_size);
        let mut biases = Matrix::new(output_size, 1);

        // Initialize weights and biases with random values
        for i in 0..output_size {
            for j in 0..input_size {
                weights.set(i, j, normal.sample(&mut rand_gen));
            }
            biases.set(i, 0, normal.sample(&mut rand_gen));
        }

        FullLayer::new(weights, biases, act_func)
    }

    pub fn forward(&self, input: &Matrix<f64>) -> Matrix<f64> {
        // Perform matrix multiplication and add bias
        let mut output = self.weights.clone() * input.clone() + self.biases.clone();
        
        // Apply activation function
        output.apply_to_all(&|x| self.act_func.evaluate(x));
        
        output
    }
}

//#[derive(Clone)]
//pub struct ConvNeuralNet {
//    pub layers: Vec<Layer>,
//}

