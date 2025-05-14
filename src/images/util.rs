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
    use super::{read_rgba_matrices, write_rgba_matrices};


    #[test]
    fn test_readwrite() {
        let channels = read_rgba_matrices("testing/files/sheep.png");
        write_rgba_matrices(channels, "testing/files/sheep_new.png");
    }

    #[test]
    fn test_compression() {
        let channels = read_rgba_matrices("testing/files/sheep.png");
    }

}