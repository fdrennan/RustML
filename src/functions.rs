use std::path::Path;
use std::fs::File;
use std::convert::TryFrom;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
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

pub fn feature_and_target(in_df: &DataFrame, target_column: &str) -> (PolarResult<DataFrame>,PolarResult<DataFrame>) {

    let target = in_df.select(target_column);
    let predictors = in_df.drop(target_column);
    let predictor_names = predictors.as_ref().unwrap().get_column_names();
    println!("Predicting on {:?}", target_column);
    println!("...using features {:?}", predictor_names);
    (predictors, target)

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
