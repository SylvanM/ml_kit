use matrix_kit::dynamic::matrix::Matrix;

pub struct Perceptron {
    weight: Matrix<f64>,

    bias: Matrix<f64>,
    learning_rate: f64,
}

impl Perceptron {

    fn check_invariant(&self) {

        debug_assert_eq!(self.weight.row_count(), 1);
        debug_assert_eq!(self.bias.row_count(), 1);
        debug_assert_eq!(self.bias.col_count(), 1);
    }

    /// MARK: Constructors

    pub fn new(weight: Matrix<f64>, bias: Matrix<f64>, learning_rate: f64) -> Self {
        let p = Perceptron {
            weight,
            bias,
            learning_rate,
        };
        p.check_invariant();
        p
    }

    /// MARK: Methods

    pub fn predict(&self, x: &Matrix<f64>) -> Matrix<f64> {
        debug_assert_eq!(x.row_count(), self.weight.col_count());
        debug_assert_eq!(x.col_count(), 1);

        let dot = self.weight.clone() * x.clone();    // (1 x d) * (d x 1) => (1 x 1)
        let output = dot + self.bias.clone(); // (1 x 1) + (1 x 1) => (1 x 1)
        output
    }

    /// Mark: Getters

    pub fn weight(&self) -> &Matrix<f64> {
        &self.weight
    }

    pub fn bias(&self) -> &Matrix<f64> {
        &self.bias
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
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
            for i in 0..x.col_count() {
                let x_i = Matrix::from_flatmap(x.row_count(), 1, 
                    (0..x.row_count()).map(|r| x.get(r, i)).collect());
                let y_i = Matrix::from_flatmap(1, 1, vec![y.get(0, i)]);

                let prediction = self.predict(&x_i);
                let error = y_i - prediction;

                // Update weights and bias
                let x_t = x_i.transpose();
                let delta_w = error.clone() * x_t;
                
                self.weight = self.weight.clone() + (delta_w * self.learning_rate);
                self.bias = self.bias.clone() + (error * self.learning_rate);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_kit::dynamic::matrix::Matrix;

    #[test]
    fn test_perceptron_construction() {
        let w = Matrix::from_flatmap(1, 2, vec![0.0, 0.0]);
        let b = Matrix::from_flatmap(1, 1, vec![0.0]);
        let p = Perceptron::new(w, b, 0.1);
        
        assert_eq!(p.weight().row_count(), 1);
        assert_eq!(p.weight().col_count(), 2);
        assert_eq!(p.bias().row_count(), 1);
        assert_eq!(p.bias().col_count(), 1);
    }

    #[test]
    fn test_perceptron_predict() {
        // Create a perceptron that implements AND gate
        let w = Matrix::from_flatmap(1, 2, vec![1.0, 1.0]);
        let b = Matrix::from_flatmap(1, 1, vec![-1.5]);
        let p = Perceptron::new(w, b, 0.1);

        // Test AND gate inputs
        let inputs = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 0.0),
            (vec![1.0, 0.0], 0.0),
            (vec![1.0, 1.0], 1.0),
        ];

        for (input, expected) in inputs {
            let x = Matrix::from_flatmap(2, 1, input);
            let output = p.predict(&x);
            let result = if output.get(0, 0) > 0.0 { 1.0 } else { 0.0 };
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_perceptron_setters() {
        let w = Matrix::from_flatmap(1, 2, vec![0.0, 0.0]);
        let b = Matrix::from_flatmap(1, 1, vec![0.0]);
        let mut p = Perceptron::new(w, b, 0.1);

        // Test weight setter
        let new_w = Matrix::from_flatmap(1, 2, vec![1.0, 1.0]);
        p.set_weight(new_w.clone());
        assert_eq!(p.weight(), &new_w);

        // Test bias setter
        let new_b = Matrix::from_flatmap(1, 1, vec![1.0]);
        p.set_bias(new_b.clone());
        assert_eq!(p.bias(), &new_b);

        // Test learning rate setter
        p.set_learning_rate(0.5);
        assert_eq!(p.learning_rate(), 0.5);
    }
}

