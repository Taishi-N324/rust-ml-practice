use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// KNN
use smartcore::math::distance::Distances;
use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
// Model performance
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;

fn main() {
    // Load dataset
    let cancer_data = boston::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        cancer_data.num_samples,
        cancer_data.num_features,
        &cancer_data.data,
    );
    // These are our target class labels
    let y = cancer_data.target;
    // Split dataset into training/test (80%/20%)
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    // KNN regressor
    let y_hat_knn = KNNRegressor::fit(
        &x_train,
        &y_train,    
        KNNRegressorParameters::default().with_distance(Distances::euclidian()),
    ).and_then(|knn| knn.predict(&x_test)).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_knn));

}
