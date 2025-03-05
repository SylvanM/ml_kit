use matrix_kit::dynamic::matrix::Matrix;

pub struct Perceptron {
    /// weight vector
    weight: Matrix<f64>,

    /// bias term, 1 * 1 matrix for consistency
    bias: Matrix<f64>,
    learning_rate: f64,
}

impl Perceptron {

    /// Checks that this Perceptron is well-formed
    fn check_invariant(&self) {

        debug_assert_eq!(self.weight.row_count(), 1);
        debug_assert_eq!(self.bias.row_count(), 1);
        debug_assert_eq!(self.bias.col_count(), 1);
    }

    /// MARK: Constructors 

    pub fn new(weight: Matrix<f64>, bias: Matrix<f64>) { 

        let p = Perceptron { weight, bias, learning_rate}; 
        p.check_invariant(); 
        p 
    }

    /// MARK: Methods 

    pub fn predict(&self, x: &Matrix<f64>) {

        debug_assert_eq!(x.row_count(), self.weight.col_count());
        debug_assert_eq!(x.col_count(), 1);

        let dot = &self.weight * x;    // (1 x d) * (d x 1) => (1 x 1)
        let output = &dot + &self.bias; // (1 x 1) + (1 x 1) => (1 x 1)
        output
    
    }

    /// Mark: Getters 

    pub fn weight(&self) -> &Matrix<f64> { 
        &self.weight
    }

    pub fn bias(&self) -> &Matrix<f64> { 
        &self.bias
    }

    pub fn learning_rate(&self) -> &Matrix<f64> { 
        &self.learning_rate
    }


    /// Mark: Setters 
    pub fn set_weight(&mut self, new_weight: Matrix<f64>) {
        self.weight = new_weight;
        self.check_invariant();
    }

    pub fn set_bias(&mut self, new_bias: Matrix<f64>) {
        self.bias = new_bias;
        self.check_invariant();
    }

    pub fn set_learning_rate(&mut self, new_lr: f64) {
        self.learning_rate = new_lr;
    }


    pub fn train(&mut self, x: &Matrix<f64>, y: &Matrix<f64>, epochs: usize) {
        for _epoch in 0..epochs {
            let prediction = self.predict(x); 
            // error = y - prediction
            let error = y - &prediction; 

            // weight update w := w + lr * error * x^T
            let x_t = x.transpose();    
            let delta_w = &error * &x_t; 

            self.weight = &self.weight + &(delta_w * self.learning_rate);

            self.bias = &self.bias + &(&error * self.learning_rate);
        }
    }
}

mod tests {
    use super::*;
    use matrix_kit::dynamic::matrix::Matrix;

    #[test]
    fn test_perceptron_train() {
        let w = Matrix::from_array(&[[0.0, 0.0]]);
        let b = Matrix::from_array(&[[0.0]]);
        let mut p = Perceptron::new(w, b, 0.1);

        // Suppose we want to train on a single data point: x = [1.0, 1.0], label = 1.0
        let x = Matrix::from_array(&[[1.0], [1.0]]); // 2 x 1
        let y = Matrix::from_array(&[[1.0]]);        // 1 x 1

        p.train(&x, &y, 10);

        let pred_label = p.predict_label(&x);
    }
}