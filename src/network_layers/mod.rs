use std::process;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_rand::RandomExt;
use rand_distr::{Normal, Binomial, Distribution};

#[derive(Debug)]
/// The individual layer for the nerual network. 
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub outputs: Option<Array2<f64>>,
    pub inputs: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
    pub dweights: Option<Array2<f64>>,
    pub dbiases: Option<Array2<f64>>,
    pub weight_momentums: Option<Array2<f64>>,
    pub bias_momentums: Option<Array2<f64>>,
    pub weight_cache: Option<Array2<f64>>,
    pub bias_cache: Option<Array2<f64>>,
    pub weight_regularizer_l1: f64,
    pub weight_regularizer_l2: f64,
    pub bias_regularizer_l1: f64,
    pub bias_regularizer_l2: f64,
}

impl DenseLayer {
    pub fn new(
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
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.outputs = Some(inputs.dot(&self.weights) + &self.biases);
        let inputs = inputs.map(|x| *x);
        self.inputs = Some(inputs);
    }
    pub fn backward(&mut self, dvalues: &Array2<f64>) {
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
/// An individual layer for the neural network that only passes a certain amount of information to the next layer.
pub struct DropoutLayer {
    pub rate: f64,
    pub inputs: Option<Array2<f64>>,
    pub outputs: Option<Array2<f64>>,
    pub binary_mask: Option<Array2<f64>>,
    pub dinputs: Option<Array2<f64>>,
}

impl DropoutLayer {
    pub fn new(rate: f64) -> DropoutLayer {
        DropoutLayer {
            rate: (1.0 - rate),
            inputs: None,
            outputs: None,
            binary_mask: None,
            dinputs: None,
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
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

    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        // Gradient on values
        self.dinputs = Some(dvalues * self.binary_mask.as_ref().unwrap());
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