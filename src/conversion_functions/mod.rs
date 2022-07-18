//! conversion_functions
//! 
//! This module is a collection of diffrent functions that convert a variety of vectors and arrays to proper inputs for `ksnn`'s networks.

use std::process;
use ndarray::Array;
use ndarray::Array2;

/// Converts a vector full of i32 values into a 2d array that is one-hot encoded to properly use with ksnn nueral networks.
/// Requires a string vector as input and the number of classes in the anwser data that the network will be trained with for
/// proper conversion.
pub fn vec_i32_to_one_hot(vector: Vec<i32>, num_classes: usize) -> Array2<i32> {
    vec_i32_to_one_hot_error_handle(vector, num_classes).unwrap_or_else(|err| {
        println!("Error in one-hot encoding conversion: {}", err);
        process::exit(1);
    })
}

fn vec_i32_to_one_hot_error_handle(vector: Vec<i32>, num_classes: usize) -> Result<Array2<i32>, &'static str> {
    let mut temp: Array2<i32> = Array::zeros((vector.len(), num_classes));

    for outer in 0..temp.shape()[0] {
        for inner in 0..temp.shape()[1] {
            if vector[outer] as usize == inner {
                temp[(outer, inner as usize)] = 1;
            }
        }
    }
    let y = temp;
    Ok(y)
}

/// Converts a 1d array full of i32 values into a 2d array that is one-hot encoded to properly use with ksnn nueral networks.
/// Requires a string vector as input and the number of classes in the anwser data that the network will be trained with for
/// proper conversion.
pub fn array_i32_to_one_hot(array: Array<i32, ndarray::Dim<[usize; 1]>>, num_classes: usize) -> Array2<i32> {
    array_i32_to_one_hot_error_handle(array, num_classes).unwrap_or_else(|err| {
        println!("Error in one-hot encoding conversion: {}", err);
        process::exit(1);
    })
}

fn array_i32_to_one_hot_error_handle(array:Array<i32, ndarray::Dim<[usize; 1]>>, num_classes: usize) -> Result<Array2<i32>, &'static str> {
    let mut temp: Array2<i32> = Array::zeros((array.len(), num_classes));
    for outer in 0..temp.shape()[0] {
        for inner in 0..temp.shape()[1] {
            if array[outer] as usize == inner {
                temp[(outer, inner as usize)] = 1;
            }
        }
    }
    let y = temp;
    Ok(y)
}

/// Converts a 1d array full of f64 values into a 2d array that is one-hot encoded to properly use with ksnn nueral networks.
/// Requires a string vector as input and the number of classes in the anwser data that the network will be trained with for
/// proper conversion.
pub fn array_f64_to_one_hot(array: Array<f64, ndarray::Dim<[usize; 1]>>, num_classes: usize) -> Array2<i32> {
    array_f64_to_one_hot_error_handle(array, num_classes).unwrap_or_else(|err| {
        println!("Error in one-hot encoding conversion: {}", err);
        process::exit(1);
    })
}

fn array_f64_to_one_hot_error_handle(array:Array<f64, ndarray::Dim<[usize; 1]>>, num_classes: usize) -> Result<Array2<i32>, &'static str> {
    let mut temp: Array2<i32> = Array::zeros((array.len(), num_classes));
    for outer in 0..temp.shape()[0] {
        for inner in 0..temp.shape()[1] {
            if array[outer] as usize == inner {
                temp[(outer, inner as usize)] = 1;
            }
        }
    }
    let y = temp;
    Ok(y)
}