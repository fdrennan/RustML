mod functions;
use functions::{read_csv, feature_and_target, convert_features_to_matrix};
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
use polars::prelude::*;

fn main() {
    let file_path = "boston_dataset.csv";
    let target = "chas";

    let df = read_csv(&file_path).unwrap();
    let (features, target) = feature_and_target(&df, target);
    let x_matrix = convert_features_to_matrix(&features.unwrap());
    let target_array = target.unwrap().to_ndarray::<Float64Type>().unwrap();

    let mut y: Vec<f64> = Vec::new();
    for val in target_array.iter(){
        y.push(*val);
    }

    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x_matrix.unwrap(), &y, 0.3, true);

    let linear_regression = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    let predictions = linear_regression.predict(&x_test).unwrap();
    let mse = mean_squared_error(&y_test, &predictions);
    println!("MSE: {:?}", mse);
}