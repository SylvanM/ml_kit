use std::{fmt::Debug, iter};

use matrix_kit::dynamic::matrix::Matrix;
use rand::seq::SliceRandom;
use std::cmp::min;

/// A training sample that has a correct input and output
pub trait DataItem: Debug + Clone {

    /// The input for this training item 
    fn input(&self) -> Matrix<f64>;

    /// The correct output for this training item
    fn correct_output(&self) -> Matrix<f64>;

    /// A title, or name, of this training item
    fn name(&self) -> String;

    /// The label (for classification) corresponding to this training item
    fn label(&self) -> usize;

}

/// A dataset type, which for now is really just a queue of 
/// DataItems to output
pub struct DataSet<T: DataItem> {

    /// Just a list of all the training items. This is fine for small sets,
    /// but for larger datasets we will need to get more sophistocated 
    /// (like storing in files instead of in RAM)
    pub data_items: Vec<T>

}

impl<T: DataItem> DataSet<T> {

    /// Creates a dataset from a long list of data items 
    pub fn from_items(data_items: Vec<T>) -> Self {
        DataSet { data_items }
    }

    pub fn minibatches(&self, size: usize, max_number: usize) -> impl Iterator<Item = Vec<T>> + '_ {
        let mut rand_gen = rand::rng();
        let mut indices = vec![0 ; self.data_items.len()];
        for i in 0..indices.len() {
            indices[i] = i;
        }
        indices.shuffle(&mut rand_gen);

        let mut next_starting_index = 0;
        let mut batch_count = 0;

        let iter = iter::from_fn(move || {
            if batch_count >= max_number || next_starting_index >= self.data_items.len() {
                None
            } else {
                let batch_indices = indices[next_starting_index..(min(next_starting_index + size, self.data_items.len()))].to_vec();
                next_starting_index += size;

                let batch: Vec<T> = batch_indices.iter().map(|i| self.data_items[*i].clone()).collect();
                batch_count += 1;
                Some(batch)
            }
        });

        iter
    }

    /// Randomly sorts the dataset into batches
    pub fn all_minibatches(&self, size: usize) -> impl Iterator<Item = Vec<T>> + '_ {
        self.minibatches(size, self.data_items.len() / size + 1)
    }

    /// Randomly sample a certain number of data items 
    pub fn random_sample(&self, number: usize) -> Vec<T> {
        let mut rand_gen = rand::rng();
        let mut indices = vec![0 ; self.data_items.len()];
        for i in 0..indices.len() {
            indices[i] = i;
        }
        indices.shuffle(&mut rand_gen);

        indices[0..number].iter().map(|&i| self.data_items[i].clone()).collect()
    }

}

#[cfg(test)]
mod dataset_tests {
    use super::{DataItem, DataSet};

    impl DataItem for usize {

        fn input(&self) -> matrix_kit::dynamic::matrix::Matrix<f64> {
            todo!()
        }
    
        fn correct_output(&self) -> matrix_kit::dynamic::matrix::Matrix<f64> {
            todo!()
        }
    
        fn name(&self) -> String {
            todo!()
        }

        fn label(&self) -> usize {
            *self
        }
    }

    #[test]
    pub fn batch_test() {
        let dataset = DataSet::from_items(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

        for batch in dataset.all_minibatches(4) {
            println!("Batch: {:?}", batch);
        }
    }

}