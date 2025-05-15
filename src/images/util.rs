//!
//! A collection of utility functions for dealing with images
//! 

use image::{GenericImageView, ImageBuffer, Pixel, Rgba};
use matrix_kit::dynamic::matrix::Matrix;

/// Reads an image file and decomposes the contents into 4 matrices, each 
/// representing the different color channels of the image, R, G, B, and A.
/// 
/// This represents each channel as being on a scale from 0 to 1, NOT 0 to 255.
pub fn read_rgba_matrices(
    path: &str
) -> Vec<Matrix<f64>> {

    let picture = image::open(path)
        .expect(&format!("Unable to open image at path {:?}", path));

    let width = picture.width() as usize;
    let height = picture.height() as usize;

    let mut rgba_raw = vec![
        Matrix::<u8>::from_index_def(height, width, &mut |_,_| 0) ; 4
    ];

    for x in 0..width {
        for y in 0..height {
            let pixel = picture.get_pixel(x as u32, y as u32);
            
            for i in 0..4 {
                rgba_raw[i].set(y, x, pixel.channels()[i]);
            }
        }
    }

    rgba_raw.iter().map(|byte_matrix|
        byte_matrix.applying_to_all(&mut |b|
            (b as f64) / (u8::MAX as f64)
        )
    ).collect()
}

/// Writes R, G, B, and A channels as an image to a specific path
pub fn write_rgba_matrices(rgba: Vec<Matrix<f64>>, path: &str) {

    // Assuming all channels have the same dimensions
    let height = rgba[0].row_count();
    let width = rgba[0].col_count();

    // Create a new image buffer
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

    // Iterate over each pixel and set the values from the RGBA channels
    for y in 0..height {
        for x in 0..width {

            let mut rgba_vec = [0 ; 4];

            for i in 0..4 {
                rgba_vec[i] = (rgba[i].get(y, x) * (u8::MAX as f64)) as u8;
            }

            img_buffer.put_pixel(x as u32, y as u32, Rgba(rgba_vec));
        }
    }

    // Save the image buffer to a file
    img_buffer
        .save(path)
        .expect("Failed to save image");

}

#[cfg(test)]
mod test_image_util {
    use std::fs::File;

    use matrix_kit::dynamic::matrix::Matrix;

    use crate::{math::{activation::AFI, svd::{compressed_svd, svd, truncate}}, models::neuralnet::NeuralNet};

    use super::{read_rgba_matrices, write_rgba_matrices};


    #[test]
    fn test_readwrite() {
        let channels = read_rgba_matrices("testing/files/sheep.png");
        write_rgba_matrices(channels, "testing/files/sheep_new.png");
    }

    #[test]
    fn factor_image() {
        let channels = super::read_rgba_matrices("testing/files/sheep.png");

        // let channels: Vec<Matrix<f64>> = (0..4).map(|_|
        //     Matrix::random_normal(3, 4, 0.0, 1.0)
        // ).collect();

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
            let path = format!("testing/files/rgba_sheep_{}.mlk_nn", i);
            let mut file = match File::create(path) {
                Ok(f) => f,
                Err(e) => panic!("Error opening file: {:?}", e),
            };

            network.write_to_file(&mut file);
        }
    }

    #[test]
    fn test_compression() {
        let compressed_sizes = vec![240, 200, 150, 100, 10];
        let recovered_rgba: Vec<(Matrix<f64>, Matrix<f64>, Matrix<f64>)> = (0..4).map(|i| {
                let path = format!("testing/files/rgba_sheep_{}.mlk_nn", i);
                let mut file = match File::open(path) {
                    Ok(f) => f,
                    Err(e) => panic!("Error writing file: {:?}", e),
                };
                let network = NeuralNet::from_file(&mut file);
                let v = network.weights[0].transpose();
                let s = network.weights[1].clone();
                let u = network.weights[2].clone();

                (u, v, s)
            }  
        ).collect();

        for r in compressed_sizes {
            let truncated_rgba = recovered_rgba.iter().map(|(u, v, s)| {
                    let (u_r, v_r, s_r) = truncate(u, v, &s.get_diagonal(), r);
                    u_r * Matrix::from_diagonal(s_r) * v_r.transpose()
                }
            ).collect();

            let path = format!("testing/files/compressed_sheep_{}.png", r);
            write_rgba_matrices(truncated_rgba, &path);
        }
    }

}