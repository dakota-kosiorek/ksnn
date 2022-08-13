// Author: Dakota Kosiorek
// Date made: May 26, 2022
// Last modified: July 4, 2022
// Description: Makes and trains a nueral networks to identify intron
// and exon sequences in an overall DNA sequence.

use std::fs;
use std::path::Path;
//use ksnn::conversion_functions::array_i32_to_one_hot;
use ksnn::conversion_functions::array_f64_to_one_hot;
use ksnn::get_num_classes;
use ksnn::{ ClassificationNetwork, enable_dropout_layers };
use ksnn::optimizers::optimizer_adam;
use ndarray::Array;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use std::time::Instant;

fn main() {
    // ----------------------------------------------------------------------------------------------------
    // ---------------------------------------- Initializations -------------------------------------------
    env_logger::init();

    // ----------------------------------------------------------------------------------------------------
    // ------------------------------------------ Network Data --------------------------------------------
    
    // Data from https://pjreddie.com/projects/mnist-in-csv/
    /*
    let (x_train, y_train) = get_data("networks/MNIST/mnist_train.csv").unwrap();
    let (x_test, y_test) = get_data("networks/MNIST/mnist_test.csv").unwrap();

    let y_train = array_i32_to_one_hot(y_train, 10);
    let y_test = array_i32_to_one_hot(y_test, 10);
    */

    let (x_train, y_train) = spiral_data(500, 3);
    let (x_test, y_test) = spiral_data(100, 3);

    let y_train = array_f64_to_one_hot(y_train, 3);
    let y_test = array_f64_to_one_hot(y_test, 3);

    // ----------------------------------------------------------------------------------------------------
    // ------------------------------------------ Execute Program -----------------------------------------

    let mut neural_network = ClassificationNetwork::new(
        vec!["ActivationReLU", "SoftmaxLossCC"],
        vec![128, get_num_classes(&y_train)],
        enable_dropout_layers(true),
        optimizer_adam(0.02, 5e-7, 1e-7, 0.9, 0.99),
        &x_train,
    );

    neural_network.dropout_layers[0].rate = 0.8;
    neural_network.dense_layers[0].weight_regularizer_l2 = 5e-4;
    neural_network.dense_layers[0].bias_regularizer_l2 = 5e-4;
    
    let now = Instant::now();

    neural_network.print_epoch = false;
    neural_network.fit(200, 100, x_train, y_train, true);
    neural_network.validate(x_test, y_test, true);
    //neural_network.save("networks/MNIST/mnist_network.json").unwrap();

    println!("{}", now.elapsed().as_secs());

    //pollster::block_on(ksnn::gpu_compute::run());
}

// --------------------------------------------------------------------------------------

/// Reads the lines from a file and adds them into a string vector.
fn _get_data(filename: &str) -> Result<(Array2<f64>,  Array<i32, ndarray::Dim<[usize; 1]>>), &'static str> {
    if Path::new(&filename).exists() == false {
        return Err("file not found");
    }
    
    let contents = fs::read_to_string(&filename)
        .expect("could not read file");

    println!("Getting data from '{}'...", filename);

    let mut x_data: Array2<f64> = Array::zeros((contents.lines().count(), 784));
    let mut y_data: Array<i32, ndarray::Dim<[usize; 1]>> = Array::zeros(contents.lines().count());
    let mut current_iteration = 0;

    for line in contents.lines() {
        let text = line.to_string().replace(" ", "");
        let mut new_line: Vec<&str> = text.split(",").collect();
        let current_y = new_line[0];
        new_line.remove(0);

        y_data[current_iteration] = current_y.parse::<i32>().unwrap();

        for inner in 0..784 {
            x_data[(current_iteration, inner)] = new_line[inner].parse::<f64>().unwrap();
        }

        current_iteration += 1;
    }
    
    Ok((x_data, y_data))
}

// ----------------------------------------------------------------------------------------------------
// ----------------------------------------- Generate Data --------------------------------------------

type X = Array<f64, ndarray::Dim<[usize; 2]>>;
type Y = Array<f64, ndarray::Dim<[usize; 1]>>;

fn spiral_data(points: usize, classes: usize) -> (X, Y) {
    let mut y: ndarray::Array<f64, ndarray::Dim<[usize; 1]>> = Array::zeros(points * classes);
    let mut x = Vec::with_capacity(points * classes * 2);

    for class_number in 0..classes {
        let r = Array::linspace(0.0, 1.0, points);
        let t = (Array::linspace(
            (class_number * 4) as f64,
            ((class_number + 1) * 4) as f64,
            points,
        ) + Array::random(points, Normal::new(0.0, 1.0).unwrap()) * 0.2)
            * 2.5;
        let r2 = r.clone();
        let mut c = Vec::<f64>::new();
        for (x, y) in (r * t.map(|x| (x).sin()))
            .into_raw_vec()
            .iter()
            .zip((r2 * t.map(|x| (x).cos())).into_raw_vec().iter())
        {
            c.push(*x);
            c.push(*y);
        }
        for (ix, n) in
            ((points * class_number)..(points * (class_number + 1))).zip((0..).step_by(2))
        {
            x.push(c[n]);
            x.push(c[n + 1]);
            y[ix] = class_number as f64;
        }
    }
    (
        ndarray::ArrayBase::from_shape_vec((points * classes, 2), x).unwrap(),
        y,
    )
}
