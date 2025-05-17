use std::fs::File;

use image::imageops::FilterType::Nearest;
use matrix_kit::dynamic::matrix::Matrix;
use ml_kit::math::activation::AFI;
use ml_kit::math::svd::{self, svd};
use ml_kit::models::neuralnet::NeuralNet;
//use ml_kit::training::learning_rate::FixedLR;
use ml_kit::training::learning_rate::AdaGrad;
use ml_kit::{
    math::loss::LFI, training::sgd::SGDTrainer, utility::mnist::mnist_utility::load_mnist,
};
use rand::Rng;

fn main() {
    factor_image();
}


    fn factor_image() {
        let channels = ml_kit::images::util::read_rgba_matrices("testing/files/cat.png");

        println!("Image is [{} x {}]", channels[0].row_count(), channels[0].col_count());

        let image_svd_rgba: Vec<(Matrix<f64>, Matrix<f64>, Matrix<f64>)> = channels.iter().map(|channel|
            svd(channel)
        ).collect();

        for i in 0..4 {
            let (u, v, s) = image_svd_rgba[i].clone();

            // We are going to write this as a neural network, because we already have code that can store it!

            let network = NeuralNet::new(vec![
                v.transpose(), s.clone(), u.clone()], 
                vec![
                    Matrix::new(v.col_count(), 1),
                    Matrix::new(s.row_count(), 1),
                    Matrix::new(u.row_count(), 1),
                ], 
                vec![AFI::Identity ; 3]
            );

            // Write this down!
            let path = format!("testing/files/rgba_cat_{}.mlk_nn", i);
            let mut file = match File::create(path) {
                Ok(f) => f,
                Err(e) => panic!("Error opening file: {:?}", e),
            };

            network.write_to_file(&mut file);
        }
    }

fn matrices_close(a: &Matrix<f64>, b: &Matrix<f64>) -> bool {
        if a.row_count() != b.row_count() || a.col_count() != b.col_count() {
            return false;
        } 
        
        for r in 0..a.row_count() {
            for c in 0..a.col_count() {
                if (a.get(r, c) - b.get(r, c)).abs() > 1e-5 {
                    return false;
                }
            }
        }

        return true;
    }

fn test_svd() {
    let trials = 10;
    let mut convergence_failures = 0;

    
    for i in 1..=trials {
        let mut rng = rand::rng();

        let n = rng.random_range(200..=300);
        let m = rng.random_range(200..=300);

        println!("{}\tSVD-ing [{} x {}]\t", i, m, n);

        let a = Matrix::random_normal(m, n, 0.0, 1.0);

        let (u, v, s) = svd(&a);

        if u.row_count() == 0 {
            convergence_failures += 1;
            continue;
        }

        let alleged_a = u.clone() * s.clone() * v.transpose();
        let alleged_s = u.transpose() * a.clone() * v.clone();

        assert!(matrices_close(&alleged_a, &a));
        assert!(matrices_close(&alleged_s, &s));
        assert!(matrices_close(&(u.transpose() * u), &Matrix::identity(m, m)));
        assert!(matrices_close(&(v.transpose() * v), &Matrix::identity(n, n)));


    }

    println!("Convergence failures: {}%", 100.0 * (convergence_failures as f64) / (trials as f64));
}