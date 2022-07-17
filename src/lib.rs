//! ksnn
//! 
//! `ksnn`, or Kosiorek's Simple Neural Networks, is a crate that simplifies the creation, training, and validation of a neural network. 
//! The crate is heavily inspired by "Neural Networks from Scratch in Python" by Harrison Kinsley & Daniel Kukieła.
//! 
//! # Crate TODO's
//! * Improving crate efficiency, likely through multithreading
//! * Addition of more network types, such as regression networks
//! * Addition of more activation and entropy functions

#![allow(dead_code)]
use std::process;
//use std::fs::File;
//use std::io::prelude::*;
use ndarray::Array;
use ndarray::Array2;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
//use serde::{Deserialize, Serialize};

pub mod activation_functions;
pub mod network_layers;
pub mod optimizers;
pub mod conversion_functions;

use crate::activation_functions::*;
use crate::network_layers::*;
use crate::optimizers::*;

#[derive(Debug)]
/// Stores all information needed for the training and validating of a classification focused network. The network has parameters that can be manually 
/// set such as an individual layers bias and weight l1/l2 regularizer, an individual layers dropout rate if dropout layers have been enabled, and if the 
/// training progress should be displayed as a progress bar or if the loss and accuaracy should be printed to the terminal.
/// # Examples
/// ```
/// 
/// let x_train = ndarray::arr2(&[
///     [0.7, 0.29, 1.0, 0.55, 0.33, 0.27],
///     [0.01, 0.08, 0.893, 0.14, 0.19, 0.98]
/// ]);
/// 
/// let y_train = ndarray::arr2(&[
///     [0, 0, 1],
///     [0, 1, 0]
/// ]);
/// 
/// let x_test = ndarray::arr2(&[
///     [0.64, 0.456, 0.68, 0.1, 0.123, 0.32],
///     [0.78, 0.56, 0.58, 0.12, 0.37, 0.46]
/// ]);
/// 
/// let y_test = ndarray::arr2(&[
///     [1, 0, 0],
///     [0, 1, 0]
/// ]);
/// 
/// let mut neural_network = ksnn::ClassificationNetwork::new(
///     vec!["ActivationReLU", "ActivationReLU", "ActivationReLU", "SoftmaxLossCC"],
///     vec![32, 64, 48, 3],
///     ksnn::enable_dropout_layers(true),
///     ksnn::optimizers::optimizer_adam_def(),
///     &x_train,
/// );
///
/// neural_network.fit(100, 1, x_train, y_train);
/// neural_network.validate(x_test, y_test);
/// ```
/// *Note above example would not produce tangible results due to small training and validation datasets.
/// 
/// Some parts of an individual layer in the network can be adjusted, such as the dropout rate if it was enabled at network declaration.
/// ```
/// let x_train = ndarray::arr2(&[
///     [0.7, 0.29, 1.0, 0.55, 0.33, 0.27],
///     [0.01, 0.08, 0.893, 0.14, 0.19, 0.98]
/// ]);
/// 
/// let mut neural_network = ksnn::ClassificationNetwork::new(
///     vec!["ActivationReLU", "SoftmaxLossCC"],
///     vec![32, 3],
///     ksnn::enable_dropout_layers(true),
///     ksnn::optimizers::optimizer_adam_def(),
///     &x_train,
/// );
/// 
/// neural_network.dropout_layers[0].rate = 0.85;
/// ```
/// l1 and l2 regularization can also be adjusted for an individual layer
/// ```
/// let x_train = ndarray::arr2(&[
///     [0.7, 0.29, 1.0, 0.55, 0.33, 0.27],
///     [0.01, 0.08, 0.893, 0.14, 0.19, 0.98]
/// ]);
/// 
/// let mut neural_network = ksnn::ClassificationNetwork::new(
///     vec!["ActivationReLU", "SoftmaxLossCC"],
///     vec![32, 3],
///     ksnn::enable_dropout_layers(true),
///     ksnn::optimizers::optimizer_adam_def(),
///     &x_train,
/// );
/// 
/// neural_network.dense_layers[0].weight_regularizer_l1 = 5e-4;
/// neural_network.dense_layers[0].weight_regularizer_l2 = 5e-4;
/// neural_network.dense_layers[0].bias_regularizer_l1 = 5e-4;
/// neural_network.dense_layers[0].bias_regularizer_l2 = 5e-4;
/// ```
pub struct ClassificationNetwork {
    dense_layer_activations: Vec<ActivationFunctions>,
    pub dense_layers: Vec<DenseLayer>,
    optimizer: Optimizers,
    pub progress_bar: bool,
    pub print_epoch: bool,
    pub print_validation: bool,
    pub have_dropout_layers: bool,
    pub dropout_layers: Vec<DropoutLayer>,
}

impl ClassificationNetwork {
    /// Creates a new `ClassificationNetwork` object and takes in four values. `dense_layer_activations` is a string vector that names what activation function to use for each
    /// of the networks layers. `dense_layer_units` is a usize vector that lists how many neurons each of the networks layers will have. The final value should always be the 
    /// number of classes that the network needs to classify. `have_dropout_layers` is a bool that enables dropout layer functionality for the network. `optimizer` is the object 
    /// that corrects the network by changing weights and biases of the networks neurons. `x` is a 2d array that is the input data the network will train on.
    pub fn new(
        dense_layer_activations: Vec<&str>,
        dense_layer_units: Vec<usize>,
        have_dropout_layers: bool,
        optimizer: Optimizers,
        x: &Array2<f64>,
    ) -> Self {
        ClassificationNetwork::error_handle_new(dense_layer_activations, dense_layer_units, have_dropout_layers, optimizer, x).unwrap_or_else(|err| {
            println!("{}", err);
            process::exit(1);
        })
    }

    fn error_handle_new(
        dense_layer_activations: Vec<&str>,
        dense_layer_units: Vec<usize>,
        have_dropout_layers: bool,
        optimizer: Optimizers,
        x: &Array2<f64>,
    ) -> Result<Self, &'static str> {
        let mut dense_layers: Vec<DenseLayer> = Vec::new();
        let mut activations: Vec<ActivationFunctions> = Vec::new();
        let mut dropout_layers: Vec<DropoutLayer> = Vec::new();

        if dense_layer_units.len() < 2 {
            return Err("Error: 'layer_units' vector parameter must contain at least two items.")
        }

        if dense_layer_activations.len() != dense_layer_units.len() {
            return Err("Error: inappropriate number of activation functions to network layers.")
        }

        // Input layer
        dense_layers.push(DenseLayer::new(x.shape()[1], dense_layer_units[0], 0.0, 5e-4, 0.0, 5e-4));

        if have_dropout_layers == true {
            dropout_layers.push(DropoutLayer::new(0.1));
        }
        else {
            dropout_layers.push(DropoutLayer::new(0.0));
        }

        for i in 1..(dense_layer_units.len()) {
            dense_layers.push(DenseLayer::new(dense_layer_units[i - 1], dense_layer_units[i], 0.0, 0.0, 0.0, 0.0));

            if have_dropout_layers == true {
                dropout_layers.push(DropoutLayer::new(0.1));
            }
            else {
                dropout_layers.push(DropoutLayer::new(0.0));
            }
        }

        for i in dense_layer_activations.iter() {
            if *i == "ActivationReLU" {
                activations.push(ActivationFunctions::ActivationReLU(ActivationReLU::new()))
            }
            else if *i == "SoftmaxLossCC" {
                activations.push(ActivationFunctions::SoftmaxLossCC(SoftmaxLossCC::new()))
            }
            else {
                let msg: String = format!("Error: activation function '{}' does not exist.", i);
                return Err(Box::leak(msg.into_boxed_str()))
            }
        }

        Ok(ClassificationNetwork {
            dense_layer_activations: activations,
            dense_layers: dense_layers,
            optimizer: optimizer,
            progress_bar: false,
            print_epoch: true,
            print_validation: true,
            have_dropout_layers: have_dropout_layers,
            dropout_layers: dropout_layers,
        })
    }

    /// Trains the neural network. `training_epochs` is how many times the entire dataset is passed forward and backwards through the network. `show_progess_interval` is how often
    /// the networks loss and accuracy is displayed to the user. For example, if this parameter was supplied with a `10` then every ten epochs the loss and accuracy of the network
    /// would be displayed. `x_train` is the dataset the network will be training on. `y_train` is the anwsers for `x_train`. 
    pub fn fit(&mut self, training_epochs: u64, show_progress_interval: u64, x_train: Array2<f64>, y_train: Array2<i32>) { 
        let training_progress_bar: ProgressBar = ProgressBar::new(training_epochs);

        if self.print_epoch == true {
            self.progress_bar = false;
        }
        else {
            self.progress_bar = true;
        }

        if self.progress_bar == true {
            self.print_epoch = false;
            training_progress_bar.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:42.cyan/blue} {pos:>7}/{len:7} {msg}")
                .progress_chars("##-"));
        }

        for epoch in 0..training_epochs {
            // ----------------------------------------------------------------------------------------
            // ------------------------------------- Forward Pass -------------------------------------
            
            // Perform a forward pass of training data through this layer
            self.dense_layers[0].forward(&x_train);
                
            // Perform a forward pass through activation function
            // Takes the output of dense layer here
            self.dense_layer_activations[0].forward(self.dense_layers[0].outputs.as_ref().unwrap(), &y_train);

            if self.have_dropout_layers == true {
                self.dropout_layers[0].forward(&self.dense_layer_activations[0].get_outputs());
            }
            
            for i in 1..self.dense_layers.len() {
                if self.have_dropout_layers == true {
                    self.dense_layers[i].forward(&self.dropout_layers[i - 1].outputs.as_ref().unwrap());
                    self.dense_layer_activations[i].forward(self.dense_layers[i].outputs.as_ref().unwrap(), &y_train);
                    self.dropout_layers[i].forward(&self.dense_layer_activations[i].get_outputs());
                }
                else {
                    // Perform a forward pass through the next Dense layer
                    // Takes outputs of activation function of previous layer as inputs
                    self.dense_layers[i].forward(&self.dense_layer_activations[i - 1].get_outputs());
                    self.dense_layer_activations[i].forward(self.dense_layers[i].outputs.as_ref().unwrap(), &y_train);
                }
            }
            
            let final_activation_index = self.dense_layer_activations.len() - 1;
            
            // Perform a forward pass through the next activation function
            // Takes the output of previous dense layer and training data anwsers
            self.dense_layer_activations[final_activation_index].forward(&self.dense_layers[final_activation_index].outputs.as_ref().unwrap(), &y_train);

            let final_activation_outputs = self.dense_layer_activations[final_activation_index].get_outputs().map(|x| *x);
            
            let data_loss: f64 = self.dense_layer_activations[final_activation_index].get_data_loss().unwrap();
            let mut regularization_loss: f64 = 0.0;

            // Calculate regularization penalty
            for i in self.dense_layers.iter() {
                let single_layer_regularization_loss: f64 = self.dense_layer_activations[final_activation_index].get_regularization_loss(i).unwrap_or_else(|err| {
                    println!("Error in getting regularization loss: {}", err);
                    process::exit(1);
                });

                regularization_loss += single_layer_regularization_loss;
            }

            let loss = data_loss + regularization_loss;
            let accuracy = calculate_accuracy(&final_activation_outputs, &y_train);
            
            if ((epoch + 1) % show_progress_interval == 0 || epoch == 0) && self.print_epoch == true {
                println!("epoch: {:indent$}, acc: {:.2}%, loss: {:.3} (data_loss: {:.3} reg_loss: {:.3}), lr: {:.7}", 
                epoch + 1, 
                accuracy * 100.0,
                loss,
                data_loss,
                regularization_loss,
                self.optimizer.get_current_leaning_rate(),
                indent=training_epochs.to_string().len());
            }
            else if self.print_epoch == false {
                training_progress_bar.inc(1);
            }

            // ----------------------------------------------------------------------------------------
            // ------------------------------------ Backward Pass -------------------------------------
            

            self.dense_layer_activations[final_activation_index].backward(&final_activation_outputs, &y_train);
            let final_activation_dinputs =  self.dense_layer_activations[final_activation_index].get_dinputs().map(|x| *x);
            self.dense_layers[final_activation_index].backward(&final_activation_dinputs);

            if self.have_dropout_layers == true {
                self.dropout_layers[final_activation_index - 1].backward(&self.dense_layers[final_activation_index].dinputs.as_ref().unwrap());
            }

            for i in (0..(self.dense_layers.len() - 1)).rev() {
                if self.have_dropout_layers == true {
                    self.dense_layer_activations[i].backward(&self.dropout_layers[i].dinputs.as_ref().unwrap(), &y_train);
                    self.dense_layers[i].backward(&self.dense_layer_activations[i].get_dinputs());
                    
                    if i > 0 {
                        self.dropout_layers[i - 1].backward(&self.dense_layers[i].dinputs.as_ref().unwrap());
                    }
                }
                if self.have_dropout_layers == false {
                    self.dense_layer_activations[i].backward(&self.dense_layers[i + 1].dinputs.as_ref().unwrap(), &y_train);
                    self.dense_layers[i].backward(&self.dense_layer_activations[i].get_dinputs());
                }
            }

            self.optimizer.pre_update_params();
            for i in 0..self.dense_layers.len() {
                self.optimizer.update_params(&mut self.dense_layers[i]);
            }
            self.optimizer.post_update_params();
        }
    }
    
    /// Tests the neural network on data after training to see how accurate the network is when faced with new information not found in the dataset (Note: the network does not supply
    /// this original information, it is currently  up to the user to ensure that the information passed to this function is information that the network has not really seen before).
    /// `x_test`is the validation data the network will try to process and give correct classifications for. `y_test` is the anwsers for `x_test`.
    pub fn validate(&mut self, x_test: Array2<f64>, y_test: Array2<i32>) {
        // Perform a forward pass of training data through this layer
        self.dense_layers[0].forward(&x_test);
                
        // Perform a forward pass through activation function
        // Takes the output of dense layer here
        self.dense_layer_activations[0].forward(self.dense_layers[0].outputs.as_ref().unwrap(), &y_test);
         
        for i in 1..self.dense_layers.len() {
            // Perform a forward pass through the next Dense layer
            // Takes outputs of activation function of previous layer as inputs
            self.dense_layers[i].forward(&self.dense_layer_activations[i - 1].get_outputs());
            self.dense_layer_activations[i].forward(self.dense_layers[i].outputs.as_ref().unwrap(), &y_test);
        }
        
        let final_activation_index = self.dense_layer_activations.len() - 1;
        
        // Perform a forward pass through the next activation function
        // Takes the output of previous dense layer and training data anwsers
        self.dense_layer_activations[final_activation_index].forward(&self.dense_layers[final_activation_index].outputs.as_ref().unwrap(), &y_test);

        let final_activation_outputs = self.dense_layer_activations[final_activation_index].get_outputs().map(|x| *x);
        
        let data_loss: f64 = self.dense_layer_activations[final_activation_index].get_data_loss().unwrap();
        
        let mut regularization_loss: f64 = 0.0;

        // Calculate regularization penalty
        for i in self.dense_layers.iter() {
            let single_layer_regularization_loss: f64 = self.dense_layer_activations[final_activation_index].get_regularization_loss(i).unwrap_or_else(|err| {
                println!("Error in getting regularization loss: {}", err);
                process::exit(1);
            });

            regularization_loss += single_layer_regularization_loss;
        }

        let loss = data_loss + regularization_loss;
        let accuracy = calculate_accuracy(&final_activation_outputs, &y_test);

        if self.print_validation == true {
            println!("Validation, Acc: {:.2}%, Loss: {:.3}", 
                accuracy * 100.0,
                loss, 
            );
        }
    }

    pub fn save(&mut self, _file_name: &str) -> std::io::Result<()> {
        /*let mut file = File::create("foo.txt")?;
        file.write_all(b"Hello, world!")?;*/
        Ok(())
    }

    pub fn load(_file_name: &str) -> &str{
        "loaded network!"
    }
}

/// Calculates the accuracy of the network by seeing if the networks predictions match up with the actual anwsers.
fn calculate_accuracy(input: &Array2<f64>, y_true: &Array2<i32>) -> f64 {
    let mut predictions: Array<i32, ndarray::Dim<[usize; 1]>> = Array::zeros(input.shape()[0]);
    let mut class_targets: Array<i32, ndarray::Dim<[usize; 1]>> = Array::zeros(y_true.shape()[0]);
    let mut pred_max_num: f64 = -99999999.99;
    let mut clas_max_num: i32 = -99999999;

    // Gets the highest predicted results for each sample (basically python numpys argmax on axis=1)
    for outer in 0..input.shape()[0] {
        for inner in 0..input.shape()[1] {
            if input[(outer, inner)] > pred_max_num {
                pred_max_num = input[(outer, inner)];
                predictions[outer] = inner as i32;
            }
            if y_true[(outer, inner)] > clas_max_num {
                clas_max_num = y_true[(outer, inner)];
                class_targets[outer] = inner as i32;
            }
        }
        pred_max_num = -99999999.99;
        clas_max_num = -99999999;
    }

    let mut sum: i32 = 0;

    for i in 0..predictions.len() {
        if predictions[i] == class_targets[i] {
            sum += 1;
        }
    }

    let mean = sum as f64 / predictions.len() as f64;
    let accuracy = mean;

    accuracy
}

/// Takes in a refernece to a one-hot encoded 2d array that contains anwser data, often marked as some form of y, and returns 
/// the number of classes found in that array. 
pub fn get_num_classes(y: &Array2<i32>) -> usize {
    y.shape()[1]
}

pub fn enable_dropout_layers(is_enabled: bool) -> bool {
    is_enabled
}