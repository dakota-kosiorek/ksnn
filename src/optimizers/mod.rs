use ndarray::Array;
use crate::network_layers::DenseLayer;

#[derive(Debug)]
/// A enum of pre-made optimizers for a neural network.
pub enum Optimizers {
    OptimizerSGD(OptimizerSGD),
    OptimizerAdagrad(OptimizerAdagrad),
    OptimizerRMSprop(OptimizerRMSprop),
    OptimizerAdam(OptimizerAdam),
}

impl Optimizers {
    pub fn get_current_leaning_rate(&mut self) -> f64 {
        match self {
            Optimizers::OptimizerSGD(optimizer_sgd) => optimizer_sgd.current_learning_rate,
            Optimizers::OptimizerAdagrad(optimizer_adagrad) => optimizer_adagrad.current_learning_rate,
            Optimizers::OptimizerRMSprop(optimizer_rmsprop) => optimizer_rmsprop.current_learning_rate,
            Optimizers::OptimizerAdam(optimizer_adam) => optimizer_adam.current_learning_rate,
        }
    }
    pub fn pre_update_params(&mut self) {
        match self {
            Optimizers::OptimizerSGD(optimizer_sgd) => optimizer_sgd.pre_update_params(),
            Optimizers::OptimizerAdagrad(optimizer_adagrad) => optimizer_adagrad.pre_update_params(),
            Optimizers::OptimizerRMSprop(optimizer_rmsprop) => optimizer_rmsprop.pre_update_params(),
            Optimizers::OptimizerAdam(optimizer_adam) => optimizer_adam.pre_update_params(),
        }
    }
    pub fn update_params(&mut self, layer: &mut DenseLayer) {
        match self {
            Optimizers::OptimizerSGD(optimizer_sgd) => optimizer_sgd.update_params(layer),
            Optimizers::OptimizerAdagrad(optimizer_adagrad) => optimizer_adagrad.update_params(layer),
            Optimizers::OptimizerRMSprop(optimizer_rmsprop) => optimizer_rmsprop.update_params(layer),
            Optimizers::OptimizerAdam(optimizer_adam) => optimizer_adam.update_params(layer),
        }
    }
    pub fn post_update_params(&mut self) {
        match self {
            Optimizers::OptimizerSGD(optimizer_sgd) => optimizer_sgd.post_update_params(),
            Optimizers::OptimizerAdagrad(optimizer_adagrad) => optimizer_adagrad.post_update_params(),
            Optimizers::OptimizerRMSprop(optimizer_rmsprop) => optimizer_rmsprop.post_update_params(),
            Optimizers::OptimizerAdam(optimizer_adam) => optimizer_adam.post_update_params(),
        }
    }
}

/// Creates a `Optimizers::OptimizerSGD` and takes in a learning rate, decay, and momentum parameter.
/// # Examples
/// ``` 
/// let optimizer = ksnn::optimizers::optimizer_sgd(0.3, 1e-2, 0.6);
/// ```
pub fn optimizer_sgd(learning_rate: f64, decay: f64, momentum: f64) ->  Optimizers {
    Optimizers::OptimizerSGD(OptimizerSGD::new(learning_rate, decay, momentum))
}

/// Creates a `Optimizers::OptimizerSGD` that has a built in learning rate of 1.0, built in decay of 1e-3, and built in momentum of 0.9.
/// # Examples
/// ``` 
/// let optimizer = ksnn::optimizers::optimizer_sgd_def();
/// ```
pub fn optimizer_sgd_def() ->  Optimizers {
    Optimizers::OptimizerSGD(OptimizerSGD::new(1.0, 1e-3, 0.9))
}

/// Creates a `Optimizers::OptimizerAdagrad` and takes in a learning rate, decay, and epsilon parameter.
/// # Examples
/// ``` 
/// let optimizer = ksnn::optimizers::optimizer_adagrad(0.8, 1e-4, 1e-4);
/// ```
pub fn optimizer_adagrad(learning_rate: f64, decay: f64, epsilon: f64) ->  Optimizers {
    Optimizers::OptimizerAdagrad(OptimizerAdagrad::new(learning_rate, decay, epsilon))
}

/// Creates a `Optimizers::OptimizerAdagrad` that has a built in learning rate of 1.0, built in decay of 1e-5, and built in epsilon of 1e-7.
/// # Examples
/// ``` 
/// let optimizer = ksnn::optimizers::optimizer_adagrad_def();
/// ```
pub fn optimizer_adagrad_def() ->  Optimizers {
    Optimizers::OptimizerAdagrad(OptimizerAdagrad::new(1.0, 1e-5, 1e-7))
}

/// Creates a `Optimizers::OptimizerRMSprop` and takes in a learning rate, decay, epsilon, and rho parameter.
/// # Examples
/// ``` 
/// let optimizer = ksnn::optimizers::optimizer_rms_prop(0.08, 1e-4, 1e-7, 0.999);
/// ```
pub fn optimizer_rms_prop(learning_rate: f64, decay: f64, epsilon: f64, rho: f64) ->  Optimizers {
    Optimizers::OptimizerRMSprop(OptimizerRMSprop::new(learning_rate, decay, epsilon, rho))
}

/// Creates a `Optimizers::OptimizerRMSprop` that has a built in learning rate of 0.02, built in decay of 1e-5, built in epsilon of 1e-7, and built in rho of 0.999.
/// # Examples
/// ``` 
/// let optimizer = ksnn::optimizers::optimizer_rms_prop_def();
/// ```
pub fn optimizer_rms_prop_def() ->  Optimizers {
    Optimizers::OptimizerRMSprop(OptimizerRMSprop::new(0.02, 1e-5, 1e-7, 0.999))
}

/// Creates a `Optimizers::OptimizerAdam` and takes in a learning rate, decay, epsilon, beta 1, and beta 2 parameter.
/// # Examples
/// ``` 
/// let optimizer = ksnn::optimizers::optimizer_adam(0.06, 5e-7, 1e-7, 0.9, 0.99);
/// ```
pub fn optimizer_adam(learning_rate: f64, decay: f64, epsilon: f64, beta_1: f64, beta_2: f64) ->  Optimizers {
    Optimizers::OptimizerAdam(OptimizerAdam::new(learning_rate, decay, epsilon, beta_1, beta_2))
}

/// Creates a `Optimizers::OptimizerAdam` that has a built in learning rate of 0.05, built in decay of 5e-7, built in epsilon of 1e-7, built in beta 1 of 0.9, and built in beta 2 of 0.999.
/// # Examples
/// ``` 
/// let optimizer = ksnn::optimizers::optimizer_adam_def();
/// ```
pub fn optimizer_adam_def() ->  Optimizers {
    Optimizers::OptimizerAdam(OptimizerAdam::new(0.05, 5e-7, 1e-7, 0.9, 0.999))
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
