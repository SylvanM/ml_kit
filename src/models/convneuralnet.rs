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

    /// Return the output of the layer given input list of matrices a
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
        for r in (0..self.output_rows).step_by(self.stride) {
            for c in (0..self.output_cols).step_by(self.stride) {
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
        println!("CONVERTED DOUT: {:?}", converted_dout);
        self.d_filters.push(
            (self.converted_input.clone() * converted_dout.clone())
                .columns()
                .into_iter()
                .map(|m| {
                    Matrix::from_flatmap(f_rows * f_cols, f_depth, m.as_vec())
                        .columns()
                        .into_iter()
                        .map(|f| Matrix::from_flatmap(f_rows, f_cols, f.as_vec()))
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

        println!("d_input IN BACKPROP:");

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
                println!(
                    "l = {:}; i = {:}, d_input = {:?}, new_mat = {:?}",
                    l, i, d_input_subs[l][i], new_mat
                );
                grad = grad + new_mat;
                print!("grad = {:?}", grad);
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

#[derive(Clone)]
pub struct ConvNeuralNet {
    pub layers: Vec<Layer>,
}

impl ConvNeuralNet {
    pub fn new(layers: Vec<Layer>) -> ConvNeuralNet {
        ConvNeuralNet { layers }
    }

    /// Computes network output 
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
                        pooled_output.push(pool_layer.forward(channel));
                    }
                    current_input = pooled_output;
                }
                Layer::Full(full_layer) => {
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

    /// Classifies input 
    pub fn classify(&self, input: &Vec<Matrix<f64>>) -> (usize, f64) {
        let output = self.compute_final_layer(input);
        let output_matrix = &output[0];  // FClayer is single matrix
        
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

    /// gradient of LFI w/ respect to network
    pub fn compute_gradient(&self, input: Vec<Matrix<f64>>, target: Vec<Matrix<f64>>, loss_function: &LFI) -> CNNGradient {
        
    }
}

#[cfg(test)]
mod conv_tests {
    use std::{fs::File, ops::Range, vec};

    use super::ConvLayer;
    use crate::math::activation::AFI;
    use matrix_kit::dynamic::matrix::Matrix;

    #[test]
    fn test_conv() {
        let x = vec![
            Matrix::from_flatmap(3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            Matrix::from_flatmap(3, 3, vec![9., 8., 7., 6., 5., 4., 3., 2., 1.]),
        ];
        // println!("x = {:?}", x[0].get_diagonal());
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
        let correct = vec![
            Matrix::from_flatmap(2, 2, vec![47., 56., 74., 83.]),
            Matrix::from_flatmap(2, 2, vec![22., 25., 31., 34.]),
        ];

        println!(
            "TESTING SUBMATRIX: {:?}",
            x[0].get_submatrix(Range { start: 0, end: 1 }, Range { start: 0, end: 1 })
        );

        let mut cl = ConvLayer::new(filters, biases, AFI::ReLu, 1, 0);
        let out = cl.feedforward(&x);

        println!("Correct:");
        print_mat_list(&correct);
        println!("Outputted:");
        print_mat_list(&out);
        assert_eq!(out, correct);

        let d_output = vec![
            Matrix::from_flatmap(2, 2, vec![1., 1., 1., 1.]),
            Matrix::from_flatmap(2, 2, vec![1., 1., 1., 1.]),
        ];
        let d_input = cl.backprop(&d_output);
        println!("d_input:");
        print_mat_list(&d_input);
        println!("d_filters:");
        for i in 0..cl.d_filters[0].len() {
            print_mat_list(&cl.d_filters[0][i]);
        }
        println!("d_biases: {:?}", cl.d_biases);
    }

    fn print_mat_list(mats: &Vec<Matrix<f64>>) {
        print!("Matrix List:");
        for i in 0..mats.len() {
            print!("{:?}", mats[i])
        }
        print!("\n\n");
    }
}
