// usual imports
use std::path::Path;
use std::fs::File;
use std::vec::Vec;
use std::convert::TryFrom;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
use polars::prelude::*;
use polars::prelude::{Result as PolarResult};
use polars::frame::DataFrame;
use polars::prelude::SerReader;

pub fn read_csv<P: AsRef<Path>>(path: P) -> PolarResult<DataFrame> {
    let file = File::open(path).expect("Cannot open file.");
    let df =
        CsvReader::new(file)
        .has_header(true)
        .finish();
    df
}

pub fn feature_and_target(in_df: &DataFrame, predictor_column: &str) -> (PolarResult<DataFrame>,PolarResult<DataFrame>) {

    let target = in_df.select(predictor_column);
    let features = in_df.drop(predictor_column);

    println!("Predicting on {:?}", target.as_ref().unwrap().get_column_names());
    println!("...using features {:?}", features.as_ref().unwrap().get_column_names());
    (features, target)

}

pub fn convert_features_to_matrix(in_df: &DataFrame) -> Result<DenseMatrix<f64>>{
    let nrows = in_df.height();
    let ncols = in_df.width();
    let features_res = in_df.to_ndarray::<Float64Type>().unwrap();
    let mut xmatrix: DenseMatrix<f64> = BaseMatrix::zeros(nrows, ncols);
    let mut col:  u32 = 0;
    let mut row:  u32 = 0;

    for val in features_res.iter(){
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        xmatrix.set(m_row, m_col, *val);
        if m_col==ncols-1 {
            row+=1;
            col = 0;
        } else{
            col+=1;
        }
    }
    Ok(xmatrix)
}

fn main() {
    let ifile = "boston_dataset.csv";
    let target = "chas";

    let df = read_csv(&ifile).unwrap();
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
    let preds = linear_regression.predict(&x_test).unwrap();
    let mse = mean_squared_error(&y_test, &preds);
    println!("MSE: {:?}", mse);
}