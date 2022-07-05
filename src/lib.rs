#![allow(dead_code)]
use std::process;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_rand::RandomExt;
use rand_distr::{Normal, Binomial, Distribution};
use indicatif::ProgressBar;
use indicatif::ProgressStyle;

#[derive(Debug)]
pub enum ActivationFunctions {
    ActivationReLU(ActivationReLU),
    SoftmaxLossCC(SoftmaxLossCC),
}

impl ActivationFunctions {
    fn forward(&mut self, inputs: &Array2<f64>, y_true: &Array2<i32>) {
        match self {
            ActivationFunctions::ActivationReLU(activation_re_lu) => activation_re_lu.forward(&inputs),
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => softmax_loss_cc.forward(&inputs, y_true),
        }
    }
    fn backward(&mut self, inputs: &Array2<f64>, y_true: &Array2<i32>) {
        match self {
            ActivationFunctions::ActivationReLU(activation_re_lu) => activation_re_lu.backward(&inputs),
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => softmax_loss_cc.backward(&inputs, y_true),
        }
    }
    fn get_outputs(&mut self) -> &Array2<f64> {
        match self {
            ActivationFunctions::ActivationReLU(activation_re_lu) => activation_re_lu.outputs.as_ref().unwrap(),
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => softmax_loss_cc.outputs.as_ref().unwrap(),
        }
    }
    fn get_data_loss(&mut self) -> Result<f64, &'static str> {
        match self {
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => Ok(softmax_loss_cc.data_loss.unwrap()),
            _ => return Err("Final activation function does not have a 'data_loss' value, 
                            consider using the 'SoftmaxLossCC' activation function as your final activation function."),
        }
    }
    fn get_regularization_loss(&mut self, layer: &DenseLayer) -> Result<f64, &'static str> {
        match self {
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => Ok(softmax_loss_cc.loss.regularization_loss(layer)),
            _ => return Err("Final activation function does not have a 'data_loss' value, 
                            consider using the 'SoftmaxLossCC' activation function as your final activation function."),
        }
    }
    fn get_dinputs(&mut self) -> &Array2<f64> {
        match self {
            ActivationFunctions::ActivationReLU(activation_re_lu) => activation_re_lu.dinputs.as_ref().unwrap(),
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => softmax_loss_cc.dinputs.as_ref().unwrap(),
        }
    }
}

#[derive(Debug)]
/// There are four optimizers built into the ksnn crate:
/// 
/// ```OptimizerSGD```
/// 
/// ```OptimizerAdagrad```
/// 
/// ```OptimizerRMSprop```
/// 
/// ```OptimizerAdam```
/// 
pub enum Optimizers {
    OptimizerSGD(OptimizerSGD),
    OptimizerAdagrad(OptimizerAdagrad),
    OptimizerRMSprop(OptimizerRMSprop),
    OptimizerAdam(OptimizerAdam),
}

impl Optimizers {
    fn get_current_leaning_rate(&mut self) -> f64 {
        match self {
            Optimizers::OptimizerSGD(optimizer_sgd) => optimizer_sgd.current_learning_rate,
            Optimizers::OptimizerAdagrad(optimizer_adagrad) => optimizer_adagrad.current_learning_rate,
            Optimizers::OptimizerRMSprop(optimizer_rmsprop) => optimizer_rmsprop.current_learning_rate,
            Optimizers::OptimizerAdam(optimizer_adam) => optimizer_adam.current_learning_rate,
        }
    }
    fn pre_update_params(&mut self) {
        match self {
            Optimizers::OptimizerSGD(optimizer_sgd) => optimizer_sgd.pre_update_params(),
            Optimizers::OptimizerAdagrad(optimizer_adagrad) => optimizer_adagrad.pre_update_params(),
            Optimizers::OptimizerRMSprop(optimizer_rmsprop) => optimizer_rmsprop.pre_update_params(),
            Optimizers::OptimizerAdam(optimizer_adam) => optimizer_adam.pre_update_params(),
        }
    }
    fn update_params(&mut self, layer: &mut DenseLayer) {
        match self {
            Optimizers::OptimizerSGD(optimizer_sgd) => optimizer_sgd.update_params(layer),
            Optimizers::OptimizerAdagrad(optimizer_adagrad) => optimizer_adagrad.update_params(layer),
            Optimizers::OptimizerRMSprop(optimizer_rmsprop) => optimizer_rmsprop.update_params(layer),
            Optimizers::OptimizerAdam(optimizer_adam) => optimizer_adam.update_params(layer),
        }
    }
    fn post_update_params(&mut self) {
        match self {
            Optimizers::OptimizerSGD(optimizer_sgd) => optimizer_sgd.post_update_params(),
            Optimizers::OptimizerAdagrad(optimizer_adagrad) => optimizer_adagrad.post_update_params(),
            Optimizers::OptimizerRMSprop(optimizer_rmsprop) => optimizer_rmsprop.post_update_params(),
            Optimizers::OptimizerAdam(optimizer_adam) => optimizer_adam.post_update_params(),
        }
    }
}

pub fn optimizer_sgd(learning_rate: f64, decay: f64, momentum: f64) ->  Optimizers {
    Optimizers::OptimizerSGD(OptimizerSGD::new(learning_rate, decay, momentum))
}

pub fn optimizer_sgd_def() ->  Optimizers {
    Optimizers::OptimizerSGD(OptimizerSGD::new(1.0, 1e-3, 0.9))
}

pub fn optimizer_adagrad(learning_rate: f64, decay: f64, epsilon: f64) ->  Optimizers {
    Optimizers::OptimizerAdagrad(OptimizerAdagrad::new(learning_rate, decay, epsilon))
}

pub fn optimizer_adagrad_def() ->  Optimizers {
    Optimizers::OptimizerAdagrad(OptimizerAdagrad::new(1.0, 1e-5, 1e-7))
}

pub fn optimizer_rms_prop(learning_rate: f64, decay: f64, epsilon: f64, rho: f64) ->  Optimizers {
    Optimizers::OptimizerRMSprop(OptimizerRMSprop::new(learning_rate, decay, epsilon, rho))
}

pub fn optimizer_rms_prop_def() ->  Optimizers {
    Optimizers::OptimizerRMSprop(OptimizerRMSprop::new(0.02, 1e-5, 1e-7, 0.999))
}

/// ksnn's Adam optimizer.
pub fn optimizer_adam(learning_rate: f64, decay: f64, epsilon: f64, beta_1: f64, beta_2: f64) ->  Optimizers {
    Optimizers::OptimizerAdam(OptimizerAdam::new(learning_rate, decay, epsilon, beta_1, beta_2))
}

/// ksnn's defaults values for the Adam optimizer. The values are as follow:
/// ```no_run
/// learning_rate: 0.05
/// decay: 5e-7
/// epsilon: 1e-7
/// beta_1: 0.9
/// beta_2: 0.999
/// ```
pub fn optimizer_adam_def() ->  Optimizers {
    Optimizers::OptimizerAdam(OptimizerAdam::new(0.05, 5e-7, 1e-7, 0.9, 0.999))
}

#[derive(Debug)]
/// # Examples
/// Example 1:
/// ```no_run
/// use ksnn::ClassificationNetwork;
/// use ndarray::Array2;
/// 
/// let x: Array2<f64> = arr2(&[
///     [0.7, -0.1, 0.44],
///     [0.9, 0.26, 0.8]
/// ]);
/// let y: Array2<i32> = arr2(&[
///     [1, 0, 0, 0],
///     [0, 0, 1, 0],
/// ]);
/// 
/// let num_classes = ksnn::get_num_classes(y);
/// let nn_activation_functions = vec!["ActivationReLU", "SoftmaxLossCC"];
///let nn_units = vec![64, num_classes];
///
/// let mut neural_network = ClassificationNetwork::new(
///     &x, 
///     nn_activation_functions, 
///     nn_units, 
///     ksnn::optimizer_adam_def()
/// );
/// 
/// neural_network.dense_layers[0].weight_regularizer_l2 = 5e-4;
/// neural_network.dense_layers[0].bias_regularizer_l2 = 5e-4;
///
/// neural_network.fit(100, x_train, y_train);
/// neural_network.validate(x_test, y_test);
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
    /// Crates a new `ClassificationNetwork` object and takes in four values. Descriptions for the parameter values are:
    /// 
    /// `x`: A 2d array that is the input data the network will train on.
    /// 
    /// `dense_layer_activations`: A string vector that names what activation function to use for each
    /// of the networks dense layers.
    /// 
    /// `dense_layer_units`: A usize vector that lists how many neurons each of the networks dense layers will have.
    /// The final value should always be the number of classes that the networks needs to classify.
    /// 
    /// `optimizer`: The object that trains and corrects the network. There are four current optimizers in ksnn.
    pub fn new(
        x: &Array2<f64>,
        dense_layer_activations: Vec<&str>,
        dense_layer_units: Vec<usize>,
        optimizer: Optimizers,
        have_dropout_layers: bool,
    ) -> Self {
        ClassificationNetwork::error_handle_new(x, dense_layer_activations, dense_layer_units, optimizer, have_dropout_layers).unwrap_or_else(|err| {
            println!("{}", err);
            process::exit(1);
        })
    }

    fn error_handle_new(
        x: &Array2<f64>,
        dense_layer_activations: Vec<&str>,
        dense_layer_units: Vec<usize>,
        optimizer: Optimizers,
        have_dropout_layers: bool,
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

            for i in (0..(self.dense_layers.len() - 1)).rev() {
                self.dense_layer_activations[i].backward(&self.dense_layers[i + 1].dinputs.as_ref().unwrap(), &y_train);
                self.dense_layers[i].backward(&self.dense_layer_activations[i].get_dinputs());
            }

            self.optimizer.pre_update_params();
            for i in 0..self.dense_layers.len() {
                self.optimizer.update_params(&mut self.dense_layers[i]);
            }
            self.optimizer.post_update_params();
        }
    }
    
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
}

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

#[derive(Debug)]
/// The individual layer for the nerual network. It is made of three 2d arrays. A weights, biases, and outputs array.
/// The layer can be initizlied using the `new()` function of `DenseLayer` which take in two usize parameter.
/// The `forward()` function computes a dot product of a f64 input 2d array and `NetworkLayer`'s weights + `NetworkLayer`'s
/// biases.
pub struct DenseLayer {
    weights: Array2<f64>,
    biases: Array2<f64>,
    outputs: Option<Array2<f64>>,
    inputs: Option<Array2<f64>>,
    dinputs: Option<Array2<f64>>,
    dweights: Option<Array2<f64>>,
    dbiases: Option<Array2<f64>>,
    weight_momentums: Option<Array2<f64>>,
    bias_momentums: Option<Array2<f64>>,
    weight_cache: Option<Array2<f64>>,
    bias_cache: Option<Array2<f64>>,
    pub weight_regularizer_l1: f64,
    pub weight_regularizer_l2: f64,
    pub bias_regularizer_l1: f64,
    pub bias_regularizer_l2: f64,
}

impl DenseLayer {
    fn new(
        n_inputs: usize, 
        n_neurons: usize, 
        weight_regularizer_l1: f64, 
        weight_regularizer_l2: f64, 
        bias_regularizer_l1: f64, 
        bias_regularizer_l2: f64)
     -> Self {
        let weights = 0.01 * Array::random((n_inputs, n_neurons), Normal::new(0.0, 1.0).unwrap());
        let biases = Array::zeros((1, n_neurons));
        DenseLayer {
            weights,
            biases,
            outputs: None,
            inputs: None,
            dinputs: None,
            dweights: None,
            dbiases: None,
            weight_momentums: None,
            bias_momentums: None,
            weight_cache: None,
            bias_cache: None,
            weight_regularizer_l1: weight_regularizer_l1,
            weight_regularizer_l2: weight_regularizer_l2,
            bias_regularizer_l1: bias_regularizer_l1,
            bias_regularizer_l2: bias_regularizer_l2,
        }
    }
    fn forward(&mut self, inputs: &Array2<f64>) {
        self.outputs = Some(inputs.dot(&self.weights) + &self.biases);
        let inputs = inputs.map(|x| *x);
        self.inputs = Some(inputs);
    }
    fn backward(&mut self, dvalues: &Array2<f64>) {
        // Gradients on parameters
        self.dweights = Some(self.inputs.as_ref().unwrap().t().dot(dvalues));
        // Sums all values in the dvalues array over the 0th axis while keeping the same
        // dim as dvalue for self.dbiases
        // In python this would be np.sum(dvalues, axis=0, keepdims=True)
        let mut sum_arr_keepdims: Array2<f64> = Array::zeros((1, dvalues.shape()[1]));
        let sum_arr = dvalues.sum_axis(Axis(0));
        for i in 0..sum_arr.len() {
            sum_arr_keepdims[(0, i)] = sum_arr[i];
        }
        self.dbiases = Some(sum_arr_keepdims);

        if self.weight_regularizer_l1 > 0.0 {
            let mut d_l1: Array2<f64> = Array::ones((self.weights.shape()[0], self.weights.shape()[1]));
            for outer in 0..self.weights.shape()[0] {
                for inner in 0..self.weights.shape()[1] {
                    if self.weights[(outer, inner)] > 0.0 {
                        d_l1[(outer, inner)] = -1.0;
                    }
                }
            }
            self.dweights = Some(self.dweights.as_ref().unwrap() + (self.weight_regularizer_l1 * d_l1));
        }

        if self.weight_regularizer_l2 > 0.0 {
            self.dweights = Some(self.dweights.as_ref().unwrap() + (2.0 * self.weight_regularizer_l2 * &self.weights));
        }

        if self.bias_regularizer_l1 > 0.0 {
            let mut d_l1: Array2<f64> = Array::ones((self.biases.shape()[0], self.biases.shape()[1]));
            for outer in 0..self.biases.shape()[0] {
                for inner in 0..self.biases.shape()[1] {
                    if self.biases[(outer, inner)] > 0.0 {
                        d_l1[(outer, inner)] = -1.0;
                    }
                }
            }
            self.dbiases = Some(self.dbiases.as_ref().unwrap() + (self.bias_regularizer_l1 * d_l1));
        }

        if self.bias_regularizer_l2 > 0.0 {
            self.dbiases = Some(self.dbiases.as_ref().unwrap() + (2.0 * self.bias_regularizer_l2 * &self.biases));
        }

        // Gradient on values
        self.dinputs = Some(dvalues.dot(&self.weights.t()));
    }
}

#[derive(Debug)]
pub struct DropoutLayer {
    rate: f64,
    inputs: Option<Array2<f64>>,
    outputs: Option<Array2<f64>>,
    binary_mask: Option<Array2<f64>>,
    dinputs: Option<Array2<f64>>,
}

impl DropoutLayer {
    fn new(rate: f64) -> DropoutLayer {
        DropoutLayer {
            rate: (1.0 - rate),
            inputs: None,
            outputs: None,
            binary_mask: None,
            dinputs: None,
        }
    }

    fn forward(&mut self, inputs: &Array2<f64>) {
        let inputs_copy = inputs.map(|x| *x);
        let inputs_shape = inputs_copy.shape();
        // Save input values
        self.inputs = Some(inputs_copy.map(|x| *x));

        // Generate and save scaled mask
        let mut binary_mask: Array2<f64> = Array::zeros((1, inputs_shape[1]));
        let bin_mask: Vec<f64> = binomial(1, self.rate as f64, inputs_shape[1]);
        for i in 0..binary_mask.len() {
            binary_mask[(0, i)] = bin_mask[i] / self.rate;
        }

        let binary_mask_copy = binary_mask.map(|x| *x);
        self.binary_mask = Some(binary_mask);
        
        // Apply mask to output values
        self.outputs = Some(inputs_copy * binary_mask_copy);
    }

    fn backward(&mut self, dvalues: &Array2<f64>) {
        // Gradient on values
        self.dinputs = Some(dvalues * self.binary_mask.as_ref().unwrap());
    }
}

#[derive(Debug)]
/// An activation function used in the hidden layers of this network, 
/// the Rectified Linear Activation Function.
pub struct ActivationReLU {
    inputs: Option<Array2<f64>>,
    outputs: Option<Array2<f64>>,
    dinputs: Option<Array2<f64>>,
}

impl ActivationReLU {
    pub fn new() -> Self {
        ActivationReLU {
            inputs: None,
            outputs: None,
            dinputs: None,
        }
    }
    fn forward(&mut self, inputs: &Array2<f64>) {
        // If x <= 0, x equals 0, otherwise x = x
        let outputs = inputs.map(|x| x.max(0.0));
        let inputs = inputs.map(|x| *x);
        // Remember input values
        self.inputs = Some(inputs);
        self.outputs = Some(outputs);
    }
    fn backward(&mut self, dvalues: &Array2<f64>) {
        // drelu is a gradient with 0 as values where input values are negative
        let mut drelu: Array2<f64> = Array::zeros((dvalues.shape()[0], dvalues.shape()[1]));
        for outer in 0..drelu.shape()[0] {
            for inner in 0..drelu.shape()[1] {
                if self.inputs.as_ref().unwrap()[(outer, inner)] >= 0.0 {
                    drelu[(outer, inner)] = dvalues[(outer, inner)];
                }
            }
        }
        self.dinputs = Some(drelu);
    }
}

#[derive(Debug)]
/// The softmax actiation function is used for the output layer and 
/// returns probabilities for what each input samples classification is.
struct ActivationSoftmax {
    outputs: Option<Array2<f64>>,
    dinputs: Option<Array2<f64>>,
}

impl ActivationSoftmax {
    fn new() -> Self {
        ActivationSoftmax {
            outputs: None,
            dinputs: None,
        }
    }
    fn forward(&mut self, inputs: &Array2<f64>) {
        let mut probabilities = Array2::<f64>::zeros(inputs.raw_dim());
        for (in_row, mut out_row) in inputs.axis_iter(Axis(0)).zip(probabilities.axis_iter_mut(Axis(0))) {
            let mut max = 0.0;
            for col in in_row.iter() {
                if col > &max {
                    max = *col;
                }
            }
            // Normlaize values between 0 and 1 before being exponentialized
            let exp = in_row.map(|x| (x - max).exp());
            let sum = exp.sum();
            out_row.assign(&(exp / sum));
        }
        self.outputs = Some(probabilities);
    }
    fn backward(&mut self, dvalues: &Array2<f64>) {
        self.dinputs = Some(Array::zeros((dvalues.shape()[0], dvalues.shape()[1])));
        let arr_sh1 = dvalues.shape()[1];

        // An array the same shape as a flattened version of the self.output array
        let mut temp_output_arr: Array2<f64> = Array::zeros((arr_sh1, 1));
        let mut temp_dvalues_arr: Array2<f64> = Array::zeros((arr_sh1, 1));
        let mut l_cntr = 0;
        let mut iter_cntr = 0;

        for (_index, (single_output, single_dvalues)) in self.outputs.as_ref().unwrap().iter().zip(dvalues).enumerate() {
            temp_output_arr[(l_cntr, 0)] = *single_output;
            temp_dvalues_arr[(l_cntr, 0)] = *single_dvalues;
            l_cntr += 1;

            if l_cntr == arr_sh1 {
                let mut jacobian_matrix: Array<f64, ndarray::Dim<[usize; 2]>> = Array::zeros((temp_output_arr.shape()[0], temp_output_arr.shape()[0]));
                
                for outer in 0..jacobian_matrix.shape()[0] {
                    for inner in 0..jacobian_matrix.shape()[1] {
                        if inner == outer {
                            jacobian_matrix[(outer, inner)] = temp_output_arr[(outer, 0)];
                        }
                    }
                }

                // Calculate jacobian matrix of the output
                jacobian_matrix = jacobian_matrix - temp_output_arr.dot(&temp_output_arr.t());

                // Calculate sample-wise gradient and add it to the array of sample gradients
                let dinputs = jacobian_matrix.dot(&temp_dvalues_arr);

                // Add sample-wise gradient to the array of sample gradients
                for inner in 0..arr_sh1 {
                    self.dinputs.as_mut().unwrap()[(iter_cntr, inner)] = dinputs[(inner, 0)];
                }
                
                temp_output_arr = Array::zeros((arr_sh1, 1));
                temp_dvalues_arr = Array::zeros((arr_sh1, 1));
                l_cntr = 0;
                iter_cntr += 1;
            }
        }
    }
}

#[derive(Debug)]
struct LossCategoricalCrossentropy {
    dinputs: Option<Array2<f64>>,
    data_loss: Option<f64>,
}

impl LossCategoricalCrossentropy {
    fn new() -> Self {
        LossCategoricalCrossentropy {
            dinputs: None,
            data_loss: None,
        }
    }

    /// Calculates loss using the sparse categorical cross entropy equation.
    fn calculate(&mut self, y_pred: &Array2<f64>, y_true: &Array2<i32>) {
        let data_loss = self.forward(y_pred, y_true);
        self.data_loss = Some(data_loss);
    }

    fn forward(&mut self, y_pred: &Array2<f64>, y_true: &Array2<i32>) -> f64 {
        // clip values in the array between 1e-7 and 1-1e-7
        let y_pred = &y_pred.map(|x| x.max(0.0000001));
        let y_pred = y_pred.map(|x| x.min(0.9999999));

        let mut sum: f64 = 0.0;
        
        let correct_confidences = (y_pred * y_true.map(|x| *x as f64)).sum_axis(Axis(1));
        
        let mut negative_log_likelihoods: Array<f64, ndarray::Dim<[usize; 1]>> = Array::zeros(correct_confidences.shape()[0]);

        // Gets the sum of the negative log likelihoods for each sample
        for i in 0..correct_confidences.len() {
            negative_log_likelihoods[i] = -correct_confidences[i].ln();
            sum += negative_log_likelihoods[i];
        }

        // Calculate total data loss
        let mean = sum / y_true.shape()[0] as f64;

        let data_loss = mean;
        data_loss
    }

    fn regularization_loss(&mut self, layer: &DenseLayer) -> f64 {
        let mut regularization_loss = 0.0;

        if layer.weight_regularizer_l1 > 0.0 {
            regularization_loss += layer.weight_regularizer_l1 * (&layer.weights.map(|x| x.abs())).sum();
        }

        if layer.weight_regularizer_l2 > 0.0 {
            regularization_loss += layer.weight_regularizer_l2 * (&layer.weights * &layer.weights).sum();
        }

        if layer.bias_regularizer_l1 > 0.0 {
            regularization_loss += layer.bias_regularizer_l1 * (&layer.biases.map(|x| x.abs())).sum();
        }

        if layer.bias_regularizer_l2 > 0.0 {
            regularization_loss += layer.bias_regularizer_l2 * (&layer.biases * &layer.biases).sum();
        }

        regularization_loss
    }

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array2<i32>) {
        // Number of samples
        let samples = dvalues.shape()[0];
        // Number of labels in every sample
        let _labels = dvalues.shape()[1];
        
        // Calculate gradient
        self.dinputs = Some(-y_true.map(|x| *x as f64) / dvalues);
        // Normalize gradient
        self.dinputs = Some(self.dinputs.as_ref().unwrap() / samples as f64);
    }
}

#[derive(Debug)]
// ActivationSoftmaxLossCategoricalCrossentropy
/// A combination of the softmax activation function and the categorical loss 
/// crossentropy function that works faster.
pub struct SoftmaxLossCC {
    activation: ActivationSoftmax,
    loss: LossCategoricalCrossentropy,
    outputs: Option<Array2<f64>>,
    dinputs: Option<Array2<f64>>,
    data_loss: Option<f64>,
}

impl SoftmaxLossCC {
    pub fn new() -> Self {
        SoftmaxLossCC {
            activation: ActivationSoftmax {
                outputs: None, 
                dinputs: None,
            },
            loss: LossCategoricalCrossentropy {
                dinputs: None,
                data_loss: None,
            },
            outputs: None,
            dinputs: None,
            data_loss: None,
        }
    }
    fn forward(&mut self, inputs: &Array2<f64>, y_true: &Array2<i32>) {
        // Output layers activation function
        self.activation.forward(inputs);
        // Set the output
        let outputs = self.activation.outputs.as_ref().unwrap().map(|x| *x);
        self.outputs = Some(outputs);
        // Calculate and return loss value
        self.loss.calculate(self.outputs.as_ref().unwrap(), y_true);
        self.data_loss = self.loss.data_loss;
    }
    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array2<i32>) {
        let mut class_targets: Array<i32, ndarray::Dim<[usize; 1]>> = Array::zeros(dvalues.shape()[0]);
        let mut dinput = dvalues.map(|x| *x);
        // Gets the number of samples (or rows)
        let samples = class_targets.len();

        let mut clas_max_num: i32 = -999999999;

        // Tunrs one hot encoded values into discrete values (this loop is basically python numpys argmax on axis=1)
        for outer in 0..dinput.shape()[0] {
            for inner in 0..dinput.shape()[1] {
                if y_true[(outer, inner)] > clas_max_num {
                    clas_max_num = y_true[(outer, inner)];
                    class_targets[outer] = inner as i32;
                }
            }
            clas_max_num = -999999999;
        }

        // Calculate the gradient
        for outer in 0..dinput.shape()[0] {
            dinput[(outer, class_targets[outer] as usize)] -= 1.0;
        }

        // Normalize the gradient
        dinput = dinput / samples as f64;
        self.dinputs = Some(dinput);
    }
}

/// The networks SGD optimizer that updates layer weights and biases to 
/// reduce network loss and improve accuracy.
#[derive(Debug)]
pub struct OptimizerSGD {
    learning_rate: f64,
    current_learning_rate: f64,
    decay: f64,
    iterations: i32,
    momentum: f64,
}

impl OptimizerSGD {
    pub fn new(learning_rate: f64, decay: f64, momentum: f64) -> Self {
        OptimizerSGD {
            learning_rate: learning_rate,
            current_learning_rate: learning_rate,
            decay: decay,
            iterations: 0,
            momentum: momentum,
        }
    }
    // Called once before parameter updates
    fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations as f64));
        }
    }
    // Update parameters
    fn update_params(&mut self, layer: &mut DenseLayer) {
        // If we are using momentum for the optimizer
        if self.momentum != 0.0 {
            if layer.weight_momentums == None {
                layer.weight_momentums = Some(Array::zeros((layer.weights.shape()[0], layer.weights.shape()[1])));
                // If there is no momentum array for the assosiated weights then there is no momentum array
                // for the assosiated biases
                layer.bias_momentums = Some(Array::zeros((layer.biases.shape()[0], layer.biases.shape()[1])));
            }

            // Build weight updates with momentum - take previous
            // updates multiplied by retain factor and update with
            // current gradients
            let weight_updates = 
                self.momentum * layer.weight_momentums.as_ref().unwrap() - 
                self.current_learning_rate * layer.dweights.as_ref().unwrap();
            layer.weight_momentums = Some(weight_updates.map(|x| *x));

            // Build bias updates
            let bias_updates = 
                self.momentum * layer.bias_momentums.as_ref().unwrap() -
                self.current_learning_rate * layer.dbiases.as_ref().unwrap();
            layer.bias_momentums = Some(bias_updates.map(|x| *x));

            layer.weights = &layer.weights + weight_updates;
            layer.biases = &layer.biases + bias_updates;
        }
        // Default/Vanilla SGD updates
        else {
            layer.weights = &layer.weights + (-self.current_learning_rate * layer.dweights.as_ref().unwrap());
            layer.biases = &layer.biases + (-self.current_learning_rate * layer.dbiases.as_ref().unwrap());
        }
    }
    fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

/// The networks Adagrad optimizer that updates layer weights and biases to 
/// reduce network loss and improve accuracy.
#[derive(Debug)]
pub struct OptimizerAdagrad {
    learning_rate: f64,
    current_learning_rate: f64,
    decay: f64,
    iterations: i32,
    epsilon: f64,
}

impl OptimizerAdagrad {
    pub fn new(learning_rate: f64, decay: f64, epsilon: f64) -> Self {
        OptimizerAdagrad {
            learning_rate: learning_rate,
            current_learning_rate: learning_rate,
            decay: decay,
            iterations: 0,
            epsilon: epsilon,
        }
    }
    // Called once before parameter updates
    fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations as f64));
        }
    }
    // Update parameters
    fn update_params(&mut self, layer: &mut DenseLayer) {
        if layer.weight_cache == None {
            layer.weight_cache = Some(Array::zeros((layer.weights.shape()[0], layer.weights.shape()[1])));
            layer.bias_cache = Some(Array::zeros((layer.biases.shape()[0], layer.biases.shape()[1])));
        }
        
        // Update the weight and bias cache's with squared current gradients
        layer.weight_cache = Some(
            layer.weight_cache.as_ref().unwrap() + 
            (layer.dweights.as_ref().unwrap() * layer.dweights.as_ref().unwrap())
        );
        layer.bias_cache = Some(
            layer.bias_cache.as_ref().unwrap() + 
            (layer.dbiases.as_ref().unwrap() * layer.dbiases.as_ref().unwrap())
        );

        // Standard SGD parameter update and normalization with squared root cache
        layer.weights = &layer.weights + 
            ((-self.current_learning_rate * layer.dweights.as_ref().unwrap()) / 
            (layer.weight_cache.as_ref().unwrap().mapv(f64::sqrt) + self.epsilon));

        layer.biases = &layer.biases + 
            ((-self.current_learning_rate * layer.dbiases.as_ref().unwrap()) / 
            (layer.bias_cache.as_ref().unwrap().mapv(f64::sqrt) + self.epsilon));
    }
    fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

/// The networks RMSprop optimizer that updates layer weights and biases to 
/// reduce network loss and improve accuracy.
#[derive(Debug)]
pub struct OptimizerRMSprop {
    learning_rate: f64,
    current_learning_rate: f64,
    decay: f64,
    iterations: i32,
    epsilon: f64,
    rho: f64,
}

impl OptimizerRMSprop {
    pub fn new(learning_rate: f64, decay: f64, epsilon: f64, rho: f64) -> Self {
        OptimizerRMSprop {
            learning_rate: learning_rate,
            current_learning_rate: learning_rate,
            decay: decay,
            iterations: 0,
            epsilon: epsilon,
            rho: rho,
        }
    }
    // Called once before parameter updates
    fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations as f64));
        }
    }
    // Update parameters
    fn update_params(&mut self, layer: &mut DenseLayer) {
        if layer.weight_cache == None {
            layer.weight_cache = Some(Array::zeros((layer.weights.shape()[0], layer.weights.shape()[1])));
            layer.bias_cache = Some(Array::zeros((layer.biases.shape()[0], layer.biases.shape()[1])));
        }
        
        // Update the weight and bias cache's with squared current gradients
        layer.weight_cache = Some(
            self.rho * layer.weight_cache.as_ref().unwrap() + 
            (1.0 - self.rho) * (layer.dweights.as_ref().unwrap() * layer.dweights.as_ref().unwrap())
        );
        layer.bias_cache = Some(
            self.rho * layer.bias_cache.as_ref().unwrap() + 
            (1.0 - self.rho) * (layer.dbiases.as_ref().unwrap() * layer.dbiases.as_ref().unwrap())
        );

        // Standard SGD parameter update and normalization with squared root cache
        layer.weights = &layer.weights + 
            ((-self.current_learning_rate * layer.dweights.as_ref().unwrap()) / 
            (layer.weight_cache.as_ref().unwrap().mapv(f64::sqrt) + self.epsilon));

        layer.biases = &layer.biases + 
            ((-self.current_learning_rate * layer.dbiases.as_ref().unwrap()) / 
            (layer.bias_cache.as_ref().unwrap().mapv(f64::sqrt) + self.epsilon));
    }
    fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

/// The networks Adam optimizer that updates layer weights and biases to 
/// reduce network loss and improve accuracy.
#[derive(Debug)]
pub struct OptimizerAdam {
    learning_rate: f64,
    current_learning_rate: f64,
    decay: f64,
    iterations: i32,
    epsilon: f64,
    beta_1: f64,
    beta_2: f64,
}

impl OptimizerAdam {
    pub fn new(learning_rate: f64, decay: f64, epsilon: f64, beta_1: f64, beta_2: f64) -> Self {
        OptimizerAdam {
            learning_rate: learning_rate,
            current_learning_rate: learning_rate,
            decay: decay,
            iterations: 0,
            epsilon: epsilon,
            beta_1: beta_1,
            beta_2: beta_2,
        }
    }
    // Called once before parameter updates
    fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * 
                (1.0 / (1.0 + self.decay * self.iterations as f64));
        }
    }
    // Update parameters
    fn update_params(&mut self, layer: &mut DenseLayer) {
        if layer.weight_cache == None {
            layer.weight_momentums = Some(Array::zeros((layer.weights.shape()[0], layer.weights.shape()[1])));
            layer.weight_cache = Some(Array::zeros((layer.weights.shape()[0], layer.weights.shape()[1])));
            layer.bias_momentums = Some(Array::zeros((layer.biases.shape()[0], layer.biases.shape()[1])));
            layer.bias_cache = Some(Array::zeros((layer.biases.shape()[0], layer.biases.shape()[1])));
        }
        
        // Update the weight and bias momentums with urretn gradients
        layer.weight_momentums = Some(
            self.beta_1 * 
            layer.weight_momentums.as_ref().unwrap() + 
            (1.0 - self.beta_1) * layer.dweights.as_ref().unwrap()
        );
        layer.bias_momentums = Some(
            self.beta_1 * 
            layer.bias_momentums.as_ref().unwrap() + 
            (1.0 - self.beta_1) * layer.dbiases.as_ref().unwrap()
        );

        // Get corrected momentum
        let weight_momentums_corrected = layer.weight_momentums.as_ref().unwrap() /
            (1.0 - (self.beta_1).powf((self.iterations + 1) as f64));
        let bias_momentums_corrected = layer.bias_momentums.as_ref().unwrap() /
            (1.0 - (self.beta_1).powf((self.iterations + 1) as f64));
        
        // Update cache with squared current gradients
        layer.weight_cache = Some(
            self.beta_2 * layer.weight_cache.as_ref().unwrap() +
            (1.0 - self.beta_2) * (layer.dweights.as_ref().unwrap() * layer.dweights.as_ref().unwrap())
        );
        layer.bias_cache = Some(
            self.beta_2 * layer.bias_cache.as_ref().unwrap() +
            (1.0 - self.beta_2) * (layer.dbiases.as_ref().unwrap() * layer.dbiases.as_ref().unwrap())
        );

        // Get corrected cache
        let weight_cache_correct = layer.weight_cache.as_ref().unwrap() / 
            (1.0 - (self.beta_2).powf((self.iterations + 1) as f64));
        let bias_cache_correct = layer.bias_cache.as_ref().unwrap() / 
            (1.0 - (self.beta_2).powf((self.iterations + 1) as f64));

        // Standard SGD parameter update and normalization with squared root cache
        layer.weights = &layer.weights + 
            (-self.current_learning_rate * weight_momentums_corrected / 
            (weight_cache_correct.mapv(f64::sqrt) + self.epsilon));
            
        layer.biases = &layer.biases + 
            (-self.current_learning_rate * bias_momentums_corrected / 
            (bias_cache_correct.mapv(f64::sqrt) + self.epsilon));
    }
    fn post_update_params(&mut self) {
        self.iterations += 1;
    }
}

/// Get a vector of a binomial distribution.
///
/// `n`: number of concurrent experiments.
/// 
/// `p`: probability of the true value of the experiment.
/// 
/// `size`: how many items end up in the output vector.
fn binomial(n: u64, p: f64, size: usize) -> Vec<f64> {
    binomial_error_handle(n, p, size).unwrap_or_else(|err| {
        println!("Error in making binomial vector for dropout layer: {}", err);
        process::exit(1);
    })
}

fn binomial_error_handle(n: u64, p: f64, size: usize) -> Result<Vec<f64>, &'static str>{
    let mut bin_vec: Vec<f64> = Vec::new();
    let bin = Binomial::new(n, p).unwrap();

    for _i in 0..size {
        let v: f64 = bin.sample(&mut rand::thread_rng()) as f64;
        bin_vec.push(v);
    }
    
    Ok(bin_vec)
}

/// Converts a vector full of string values into a 2d array that is one-hot encoded to properly use with ksnn nueral networks.
/// Requires a string vector as input and the number of classes in the anwser data that the network will be trained with for
/// proper conversion.
pub fn vec_string_to_one_hot(vector: Vec<String>, num_classes: usize) -> Array2<i32> {
    vec_string_to_one_hot_error_handle(vector, num_classes).unwrap_or_else(|err| {
        println!("Error in one-hot encoding conversion: {}", err);
        process::exit(1);
    })
}

pub fn vec_string_to_one_hot_error_handle(vector: Vec<String>, num_classes: usize) -> Result<Array2<i32>, &'static str> {
    let mut data = vector.clone();
    for outer in 0..data.len() {
        data[outer] = data[outer].replace(" ", "");
        data[outer] = data[outer].replace(",", "");
    }
    let mut temp: Array2<i32> = Array::zeros((data.len(), num_classes));
    
    if num_classes != data[0].len() {
        return Err("incorrect number of classes was submitted.")
    }

    for outer in 0..temp.shape()[0] {
        for inner in 0..temp.shape()[1] {
            println!("{} {}", data[outer].chars().nth(inner).unwrap(), inner);
            if data[outer].chars().nth(inner).unwrap() as usize == inner {
                
                temp[(outer, inner as usize)] = 1;
            }
        }
    }
    let y = temp;
    Ok(y)
}

/// Converts a vector full of i32 values into a 2d array that is one-hot encoded to properly use with ksnn nueral networks.
/// Requires a string vector as input and the number of classes in the anwser data that the network will be trained with for
/// proper conversion.
pub fn vec_i32_to_one_hot(vector: Vec<i32>, num_classes: usize) -> Array2<i32> {
    vec_i32_to_one_hot_error_handle(vector, num_classes).unwrap_or_else(|err| {
        println!("Error in one-hot encoding conversion: {}", err);
        process::exit(1);
    })
}

pub fn vec_i32_to_one_hot_error_handle(vector: Vec<i32>, num_classes: usize) -> Result<Array2<i32>, &'static str> {
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

pub fn array_i32_to_one_hot_error_handle(array:Array<i32, ndarray::Dim<[usize; 1]>>, num_classes: usize) -> Result<Array2<i32>, &'static str> {
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

pub fn array_f64_to_one_hot_error_handle(array:Array<f64, ndarray::Dim<[usize; 1]>>, num_classes: usize) -> Result<Array2<i32>, &'static str> {
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

/// Takes in a refernece to a one-hot encoded 2d array that contains anwser data, often marked as some form of y, and returns 
/// the number of classes found in that array. 
pub fn get_num_classes(y: &Array2<i32>) -> usize {
    y.shape()[1]
}