use ndarray::Array;
use ndarray::Array2;
use ndarray::Axis;
use crate::network_layers::DenseLayer;

#[derive(Debug)]
/// A enum of pre-made activation functions that can be used in a neural network.
pub enum ActivationFunctions {
    ActivationReLU(ActivationReLU),
    SoftmaxLossCC(SoftmaxLossCC),
}

impl ActivationFunctions {
    pub fn forward(&mut self, inputs: &Array2<f64>, y_true: &Array2<i32>) {
        match self {
            ActivationFunctions::ActivationReLU(activation_re_lu) => activation_re_lu.forward(&inputs),
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => softmax_loss_cc.forward(&inputs, y_true),
        }
    }
    pub fn backward(&mut self, inputs: &Array2<f64>, y_true: &Array2<i32>) {
        match self {
            ActivationFunctions::ActivationReLU(activation_re_lu) => activation_re_lu.backward(&inputs),
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => softmax_loss_cc.backward(&inputs, y_true),
        }
    }
    pub fn get_outputs(&mut self) -> &Array2<f64> {
        match self {
            ActivationFunctions::ActivationReLU(activation_re_lu) => activation_re_lu.outputs.as_ref().unwrap(),
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => softmax_loss_cc.outputs.as_ref().unwrap(),
        }
    }
    pub fn get_data_loss(&mut self) -> Result<f64, &'static str> {
        match self {
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => Ok(softmax_loss_cc.data_loss.unwrap()),
            _ => return Err("Final activation function does not have a 'data_loss' value, 
                            consider using the 'SoftmaxLossCC' activation function as your final activation function."),
        }
    }
    pub fn get_regularization_loss(&mut self, layer: &DenseLayer) -> Result<f64, &'static str> {
        match self {
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => Ok(softmax_loss_cc.loss.regularization_loss(layer)),
            _ => return Err("Final activation function does not have a 'data_loss' value, 
                            consider using the 'SoftmaxLossCC' activation function as your final activation function."),
        }
    }
    pub fn get_dinputs(&mut self) -> &Array2<f64> {
        match self {
            ActivationFunctions::ActivationReLU(activation_re_lu) => activation_re_lu.dinputs.as_ref().unwrap(),
            ActivationFunctions::SoftmaxLossCC(softmax_loss_cc) => softmax_loss_cc.dinputs.as_ref().unwrap(),
        }
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
pub struct ActivationSoftmax {
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
/// Calculates the networks current total loss.
pub struct LossCategoricalCrossentropy {
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
/// Full name ActivationSoftmaxLossCategoricalCrossentropy; A combination of the softmax activation function and the categorical loss 
/// crossentropy function. The combination allows for a simplified implementation and overall faster calculation time.
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