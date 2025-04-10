use std::fmt::Debug;

use matrix_kit::dynamic::matrix::Matrix;

/// A training sample that has a correct input and output
pub trait DataItem: Debug + Clone {

    /// The input for this training item 
    fn input(&self) -> Matrix<f64>;

    /// The correct output for this training item
    fn correct_output(&self) -> Matrix<f64>;

    /// A label, or name, of this training item
    fn label(&self) -> String;

}

/// A dataset type, which for now is really just a queue of 
/// DataItems to output
#[derive(Clone)]
pub struct DataSet<T: DataItem> {

    /// Just a list of all the training items. This is fine for small sets,
    /// but for larger datasets we will need to get more sophistocated 
    /// (like storing in files instead of in RAM)
    pub data_items: Vec<T>

}
