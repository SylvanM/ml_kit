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

    fn raw_activation(&self, x: &Matrix<f64>) -> f64 {
        let dot = self.weight.inner_product(x);
        dot + self.bias.get(0, 0)
    }

    fn step(val: f64) -> f64 {
        if val > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    pub fn predict(&self, x: &Matrix<f64>) -> Matrix<f64> {
        let y_hat = Self::step(self.raw_activation(x));
        Matrix::from_flatmap(1, 1, vec![y_hat])
    }

    pub fn train(&mut self, x: &Matrix<f64>, y: &Matrix<f64>, epochs: usize) {
        let n = x.col_count();

        let mut best_w = self.weight.clone();
        let mut best_b = self.bias.clone();
        let mut best_correct = 0;

        for _ in 0..epochs {
            let mut misclassified = 0;

            for i in 0..n {
                let x_i = Matrix::from_flatmap(
                    x.row_count(),
                    1,
                    (0..x.row_count()).map(|r| x.get(r, i)).collect(),
                );
                let target = y.get(0, i);

                let raw = self.raw_activation(&x_i);
                let y_hat = Self::step(raw);
                let error = target - y_hat;

                if error != 0.0 {
                    misclassified += 1;

                    // weight update
                    let delta_w = x_i.transpose() * (self.learning_rate * error);
                    self.weight += delta_w;

                    // bias update
                    let delta_b = Matrix::from_flatmap(1, 1, vec![self.learning_rate * error]);
                    self.bias += delta_b;
                }
            }

            let correct = n - misclassified;
            if correct > best_correct {
                best_correct = correct;
                best_w = self.weight.clone();
                best_b = self.bias.clone();
            }

            if misclassified == 0 {
                break;
            }
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_kit::dynamic::matrix::Matrix;
    use rand::Rng;

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

        let new_w = Matrix::from_flatmap(1, 2, vec![1.0, 1.0]);
        p.set_weight(new_w.clone());
        assert_eq!(p.weight(), &new_w);

        let new_b = Matrix::from_flatmap(1, 1, vec![1.0]);
        p.set_bias(new_b.clone());
        assert_eq!(p.bias(), &new_b);

        p.set_learning_rate(0.5);
        assert_eq!(p.learning_rate(), 0.5);
    }

    #[test]
    fn test_2d_separable() {
        let xs = vec![-0.9, -0.8, -0.7, -0.5, 0.0, 0.2, 0.5, 0.7];
        let ys = vec![0.0, 0.0, 1.0, 1.0];

        let X = Matrix::from_flatmap(2, 4, xs);
        let Y = Matrix::from_flatmap(1, 4, ys);

        let init_w = Matrix::from_flatmap(1, 2, vec![0.0, 0.0]);
        let bias_matrix = Matrix::from_flatmap(1, 1, vec![0.0]);
        let mut p = Perceptron::new(init_w, bias_matrix, 0.01);

        p.train(&X, &Y, 1000);

        for i in 0..4 {
            let xi = Matrix::from_flatmap(2, 1, vec![X.get(0, i), X.get(1, i)]);
            assert_eq!(p.predict(&xi).get(0, 0), Y.get(0, i));
        }
    }
}
