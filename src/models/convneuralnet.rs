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
    filters: Vec<Vec<Matrix<f64>>>,
    biases: Matrix<f64>,
    act_func: AFI,
    stride: usize,
    padding: usize,
    output_rows: usize,
    output_cols: usize,
    padded_input: Vec<Matrix<f64>>,
    converted_input: Matrix<f64>,
    converted_kernel: Matrix<f64>,
    output: Vec<Matrix<f64>>,
    d_filters: Vec<Vec<Vec<Matrix<f64>>>>,
    d_biases: Vec<Matrix<f64>>,
}

impl ConvLayer {
    pub fn new(
        filters: Vec<Vec<Matrix<f64>>>,
        biases: Matrix<f64>,
        act_func: AFI,
        stride: usize,
        padding: usize,
    ) -> ConvLayer {
        ConvLayer {
            filters,
            biases,
            act_func,
            stride,
            padding,
            output_rows: 0,
            output_cols: 0,
            padded_input: Vec::new(),
            converted_input: Matrix::new(0, 0),
            converted_kernel: Matrix::new(0, 0),
            output: Vec::new(),
            d_filters: Vec::new(),
            d_biases: Vec::new(),
        }
    }

    /// Creates a new, empty convolutional layer with all filters and biases
    /// set to 0.
    pub fn zeros(
        filter_count: usize,
        filter_rows: usize,
        filter_cols: usize,
        filter_depth: usize,
        act_func: AFI,
        stride: usize,
        padding: usize,
    ) -> ConvLayer {
        let filters = (0..filter_count)
            .map(|_| {
                (0..filter_depth)
                    .map(|_| Matrix::new(filter_rows, filter_cols))
                    .collect()
            })
            .collect();

        let biases = Matrix::new(filter_count, 1);

        ConvLayer {
            filters,
            biases,
            act_func,
            stride,
            padding,
            output_rows: 0,
            output_cols: 0,
            padded_input: Vec::new(),
            converted_input: Matrix::new(0, 0),
            converted_kernel: Matrix::new(0, 0),
            output: Vec::new(),
            d_filters: Vec::new(),
            d_biases: Vec::new(),
        }
    }

    /// Generates a random convolutional layer of a particular shape
    pub fn rand(
        filter_count: usize,
        filter_rows: usize,
        filter_cols: usize,
        filter_depth: usize,
        act_func: AFI,
        stride: usize,
        padding: usize,
    ) -> ConvLayer {
        let mut rand_gen = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap(); // Tweak as needed!
        let mut conv_layer = ConvLayer::zeros(
            filter_count,
            filter_rows,
            filter_cols,
            filter_depth,
            act_func,
            stride,
            padding,
        );

        for l in 0..filter_count {
            for d in 0..filter_depth {
                for r in 0..filter_rows {
                    for c in 0..filter_cols {
                        conv_layer.filters[l][d].set(r, c, normal.sample(&mut rand_gen));
                    }
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

    /// Return the output of the layer given input list of matrices
    pub fn feedforward(&mut self, input: &Vec<Matrix<f64>>) -> Vec<Matrix<f64>> {
        let f_depth = self.filters[0].len();
        let f_rows = self.filters[0][0].row_count();
        let f_cols = self.filters[0][0].col_count();

        // Add padding if needed
        self.padded_input = input.into_iter().map(|m| self.add_padding(m)).collect();

        // Calculate output dimensions
        self.output_rows = (self.padded_input[0].row_count() - f_rows) / self.stride + 1;
        self.output_cols = (self.padded_input[0].col_count() - f_cols) / self.stride + 1;

        // Convert convolution to matrix multiplication
        let mut flatmap: Vec<f64> = Vec::new();
        for r in (0..self.output_rows * self.stride).step_by(self.stride) {
            for c in (0..self.output_cols * self.stride).step_by(self.stride) {
                for l in 0..self.padded_input.len() {
                    flatmap.append(
                        &mut self.padded_input[l]
                            .get_submatrix(
                                Range {
                                    start: r,
                                    end: r + f_rows,
                                },
                                Range {
                                    start: c,
                                    end: c + f_cols,
                                },
                            )
                            .transpose()
                            .as_vec(),
                    );
                }
            }
        }

        self.converted_input = Matrix::from_flatmap(
            f_rows * f_cols * f_depth,
            self.output_rows * self.output_cols,
            flatmap,
        );

        self.converted_kernel = Matrix::from_flatmap(
            f_rows * f_cols * f_depth,
            self.filters.len(),
            self.filters
                .clone()
                .into_iter()
                .map(|f| {
                    f.into_iter()
                        .map(|m| m.transpose().as_vec())
                        .collect::<Vec<_>>()
                        .concat()
                })
                .collect::<Vec<_>>()
                .concat(),
        )
        .transpose();

        self.output = (self.converted_kernel.clone() * self.converted_input.clone())
            .transpose()
            .columns()
            .into_iter()
            .map(|mat| {
                Matrix::from_flatmap(self.output_cols, self.output_rows, mat.as_vec()).transpose()
            })
            .collect();

        // Apply activation and add bias
        for l in 0..self.filters.len() {
            self.output[l].apply_to_all(&|x| self.act_func.evaluate(x + self.biases.get(l, 0)));
        }

        self.output.clone()
    }

    /// Compute and save layer parameter gradients and return input gradients
    pub fn backprop(&mut self, d_output: &Vec<Matrix<f64>>) -> Vec<Matrix<f64>> {
        let f_depth = self.filters[0].len();
        let f_rows = self.filters[0][0].row_count();
        let f_cols = self.filters[0][0].col_count();

        let mut d_output = d_output.clone();

        // ReLU derivative: setting d_output to 0 anywhere output is 0
        for l in 0..self.output.len() {
            for r in 0..self.output[l].row_count() {
                for c in 0..self.output[l].col_count() {
                    if self.output[l].get(r, c) == 0. {
                        d_output[l].set(r, c, 0.)
                    }
                }
            }
        }

        // Bias derivatives
        let mut d_biases_flatmap: Vec<f64> = Vec::new();
        for l in 0..self.output.len() {
            d_biases_flatmap.push(d_output[l].as_vec().iter().sum());
        }
        self.d_biases
            .push(Matrix::from_flatmap(self.output.len(), 1, d_biases_flatmap));

        // Filter derivatives
        let converted_dout = Matrix::from_flatmap(
            self.output_rows * self.output_cols,
            self.filters.len(),
            d_output
                .into_iter()
                .map(|m| m.transpose().as_vec())
                .collect::<Vec<_>>()
                .concat(),
        );

        self.d_filters.push(
            (self.converted_input.clone() * converted_dout.clone())
                .columns()
                .into_iter()
                .map(|m| {
                    Matrix::from_flatmap(f_rows * f_cols, f_depth, m.as_vec())
                        .columns()
                        .into_iter()
                        .map(|f| Matrix::from_flatmap(f_cols, f_rows, f.as_vec()).transpose())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        );

        // Input derivatives
        let d_input_subs = Matrix::from_flatmap(
            f_rows * f_cols * self.output_rows * self.output_cols,
            f_depth,
            (converted_dout.clone() * self.converted_kernel.clone()).as_vec(),
        )
        .columns()
        .into_iter()
        .map(|m| {
            Matrix::from_flatmap(
                self.output_rows * self.output_cols,
                f_rows * f_cols,
                m.as_vec(),
            )
            .transpose()
            .columns()
        })
        .collect::<Vec<_>>();

        let mut d_input: Vec<Matrix<f64>> = Vec::new();
        for l in 0..f_depth {
            let mut grad: Matrix<f64> = Matrix::new(
                self.padded_input[l].row_count(),
                self.padded_input[l].col_count(),
            );
            for i in 0..d_input_subs[l].len() {
                let r_start = i / self.output_cols * self.stride;
                let c_start = (i % self.output_cols) * self.stride;
                let mut new_mat = Matrix::new(grad.row_count(), grad.col_count());
                new_mat.set_submatrix(
                    Range {
                        start: r_start,
                        end: r_start + f_rows,
                    },
                    Range {
                        start: c_start,
                        end: c_start + f_cols,
                    },
                    Matrix::from_flatmap(f_cols, f_rows, d_input_subs[l][i].as_vec()).transpose(),
                );
                grad = grad + new_mat;
            }
            d_input.push(grad.get_submatrix(
                Range {
                    start: self.padding,
                    end: grad.row_count() - self.padding,
                },
                Range {
                    start: self.padding,
                    end: grad.col_count() - self.padding,
                },
            ));
        }

        d_input
    }
}

#[derive(Clone)]
pub enum PoolType {
    MAX,
    AVG,
    SUM,
}

#[derive(Clone)]
pub struct PoolLayer {
    pool_type: PoolType,
    w_rows: usize,
    w_cols: usize,
    stride: usize,
    input: Vec<Matrix<f64>>,
}

impl PoolLayer {
    pub fn new(pool_type: PoolType, w_rows: usize, w_cols: usize, stride: usize) -> PoolLayer {
        PoolLayer {
            pool_type,
            w_rows,
            w_cols,
            stride,
            input: Vec::new(),
        }
    }

    /// Returns the output of the layer given input list of matrices
    pub fn feedforward(&mut self, input: &Vec<Matrix<f64>>) -> Vec<Matrix<f64>> {
        self.input = input.clone();

        let output_rows = (input[0].row_count() - self.w_rows) / self.stride + 1;
        let output_cols = (input[0].col_count() - self.w_cols) / self.stride + 1;

        let mut output: Vec<Matrix<f64>> = Vec::new();
        for l in 0..input.len() {
            let mut flatmap: Vec<f64> = Vec::new();
            for c in (0..output_cols * self.stride).step_by(self.stride) {
                for r in (0..output_rows * self.stride).step_by(self.stride) {
                    let sub_mat = input[l]
                        .get_submatrix(
                            Range {
                                start: r,
                                end: r + self.w_rows,
                            },
                            Range {
                                start: c,
                                end: c + self.w_cols,
                            },
                        )
                        .as_vec();
                    flatmap.push(match self.pool_type {
                        PoolType::MAX => sub_mat.iter().cloned().fold(0. / 0., f64::max),
                        PoolType::AVG => sub_mat.iter().sum::<f64>() / (sub_mat.len() as f64),
                        PoolType::SUM => sub_mat.iter().sum(),
                    });
                }
            }
            output.push(Matrix::from_flatmap(output_rows, output_cols, flatmap));
        }

        output
    }

    /// Return input gradients
    pub fn backprop(&self, d_output: &Vec<Matrix<f64>>) -> Vec<Matrix<f64>> {
        let input_rows = self.input[0].row_count();
        let input_cols = self.input[0].col_count();

        let mut d_input: Vec<Matrix<f64>> = Vec::new();
        for l in 0..d_output.len() {
            let mut grad: Matrix<f64> = Matrix::new(input_rows, input_cols);
            for r in 0..d_output[l].row_count() {
                for c in 0..d_output[l].col_count() {
                    let mut new_mat = Matrix::new(input_rows, input_cols);
                    let row_range = Range {
                        start: r * self.stride,
                        end: r * self.stride + self.w_rows,
                    };
                    let col_range = Range {
                        start: c * self.stride,
                        end: c * self.stride + self.w_cols,
                    };
                    let mut sub_mat = Matrix::new(self.w_rows, self.w_cols);
                    match self.pool_type {
                        PoolType::MAX => {
                            let input_sub = self.input[l]
                                .get_submatrix(row_range.clone(), col_range.clone())
                                .as_vec();
                            let argmax =
                                input_sub
                                    .iter()
                                    .enumerate()
                                    .fold(0, |curr_idx, (idx, val)| {
                                        if input_sub[curr_idx] > *val {
                                            curr_idx
                                        } else {
                                            idx
                                        }
                                    });
                            sub_mat.set(argmax % self.w_cols, argmax / self.w_cols, 1.);
                        }
                        PoolType::AVG => {
                            sub_mat.apply_to_all(&|_| 1. / (self.w_rows * self.w_cols) as f64)
                        }
                        PoolType::SUM => sub_mat.apply_to_all(&|_| 1.),
                    }
                    new_mat.set_submatrix(row_range, col_range, sub_mat);
                    grad = grad + new_mat;
                }
            }
            d_input.push(grad);
        }

        d_input
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

#[derive(Clone)]
pub struct ConvNeuralNet {
    pub layers: Vec<Layer>,
}

impl ConvNeuralNet {
    pub fn new(layers: Vec<Layer>) -> ConvNeuralNet {
        ConvNeuralNet { layers }
    }

    /// Computes the output of the network for a given input
    pub fn compute_final_layer(&self, input: &Vec<Matrix<f64>>) -> Vec<Matrix<f64>> {
        let mut current_input = input.clone();
        
        for layer in &self.layers {
            match layer {
                Layer::Conv(conv_layer) => {
                    let mut conv = conv_layer.clone();
                    current_input = conv.feedforward(&current_input);
                }
                Layer::Pool(pool_layer) => {
                    let mut pooled_output = Vec::new();
                    for channel in &current_input {
                        pooled_output.push(pool_layer.feedforward(channel));
                    }
                    current_input = pooled_output;
                }
                Layer::Full(full_layer) => {
                    // Convert input channels to a single matrix for full layer
                    let mut flat_input = Matrix::new(current_input[0].row_count() * current_input[0].col_count() * current_input.len(), 1);
                    let mut i = 0;
                    for channel in &current_input {
                        for r in 0..channel.row_count() {
                            for c in 0..channel.col_count() {
                                flat_input.set(i, 0, channel.get(r, c));
                                i += 1;
                            }
                        }
                    }
                    current_input = vec![full_layer.forward(&flat_input)];
                }
            }
        }
        
        current_input
    }

    /// Classifies an input and returns the predicted class and confidence
    pub fn classify(&self, input: &Vec<Matrix<f64>>) -> (usize, f64) {
        let output = self.compute_final_layer(input);
        let output_matrix = &output[0];  // Full layer output is a single matrix
        
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = 0;
        
        for i in 0..output_matrix.row_count() {
            let val = output_matrix.get(i, 0);
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        (max_idx, max_val)
    }
}

#[cfg(test)]
mod conv_tests {
    use std::{fs::File, ops::Range, vec};

    use super::ConvLayer;
    use super::PoolLayer;
    use super::PoolType;
    use crate::math::activation::AFI;
    use matrix_kit::dynamic::matrix::Matrix;

    #[test]
    fn test_conv_layer() {
        let x = vec![
            Matrix::from_flatmap(3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            Matrix::from_flatmap(3, 3, vec![9., 8., 7., 6., 5., 4., 3., 2., 1.]),
        ];
        let filters = vec![
            vec![
                Matrix::from_flatmap(2, 2, vec![1., 2., 3., 4.]),
                Matrix::from_flatmap(2, 2, vec![1., 0., 0., 0.]),
            ],
            vec![
                Matrix::from_flatmap(2, 2, vec![1., 1., 1., 1.]),
                Matrix::from_flatmap(2, 2, vec![0., 0., 0., 1.]),
            ],
        ];
        let biases = Matrix::from_flatmap(2, 1, vec![1., 5.]);
        let correct_ff = vec![
            Matrix::from_flatmap(2, 2, vec![47., 56., 74., 83.]),
            Matrix::from_flatmap(2, 2, vec![22., 25., 31., 34.]),
        ];

        let mut cl = ConvLayer::new(filters, biases, AFI::ReLu, 1, 0);
        let out = cl.feedforward(&x);

        debug_assert_eq!(out, correct_ff);

        let d_output = vec![
            Matrix::from_flatmap(2, 2, vec![1., 1., 1., 1.]),
            Matrix::from_flatmap(2, 2, vec![2., 1., 1., 1.]),
        ];
        let correct_d_input = vec![
            Matrix::from_flatmap(3, 3, vec![3., 6., 3., 7., 15., 8., 4., 9., 5.]),
            Matrix::from_flatmap(3, 3, vec![1., 1., 0., 1., 3., 1., 0., 1., 1.]),
        ];
        let correct_d_filters = vec![vec![
            vec![
                Matrix::from_flatmap(2, 2, vec![12., 16., 24., 28.]),
                Matrix::from_flatmap(2, 2, vec![28., 24., 16., 12.]),
            ],
            vec![
                Matrix::from_flatmap(2, 2, vec![13., 18., 28., 33.]),
                Matrix::from_flatmap(2, 2, vec![37., 32., 22., 17.]),
            ],
        ]];
        let correct_d_biases = vec![Matrix::from_flatmap(2, 1, vec![4., 5.])];
        let d_input = cl.backprop(&d_output);

        debug_assert_eq!(d_input, correct_d_input);
        debug_assert_eq!(cl.d_filters, correct_d_filters);
        debug_assert_eq!(cl.d_biases, correct_d_biases);
    }

    #[test]
    fn test_pool_layer() {
        let x = vec![
            Matrix::from_flatmap(3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            Matrix::from_flatmap(3, 3, vec![9., 8., 7., 6., 5., 4., 3., 2., 1.]),
        ];

        let mut pl_max = PoolLayer::new(PoolType::MAX, 2, 2, 1);
        let out_max = pl_max.feedforward(&x);
        let out_max_correct = vec![
            Matrix::from_flatmap(2, 2, vec![5., 6., 8., 9.]),
            Matrix::from_flatmap(2, 2, vec![9., 8., 6., 5.]),
        ];
        debug_assert_eq!(out_max, out_max_correct);

        let mut pl_sum = PoolLayer::new(PoolType::SUM, 2, 2, 1);
        let out_sum = pl_sum.feedforward(&x);
        let out_sum_correct = vec![
            Matrix::from_flatmap(2, 2, vec![12., 16., 24., 28.]),
            Matrix::from_flatmap(2, 2, vec![28., 24., 16., 12.]),
        ];
        debug_assert_eq!(out_sum, out_sum_correct);

        let mut pl_avg = PoolLayer::new(PoolType::AVG, 2, 2, 1);
        let out_avg = pl_avg.feedforward(&x);
        let out_avg_correct = vec![
            Matrix::from_flatmap(2, 2, vec![3., 4., 6., 7.]),
            Matrix::from_flatmap(2, 2, vec![7., 6., 4., 3.]),
        ];
        debug_assert_eq!(out_avg, out_avg_correct)

        
    }

    fn print_mat_list(mats: &Vec<Matrix<f64>>) {
        print!("Matrix List:");
        for i in 0..mats.len() {
            print!("{:?}", mats[i])
        }
        print!("\n\n");
    }
}
