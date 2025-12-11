mod data;
mod linear;
mod model_io;
mod preprocess;
mod split;
use preprocess::StandardScaler;
use split::train_test_split;

fn main() {
    // 1. load data
    let (x, y) = data::load_csv("data.csv", 3).expect("load failed");

    // 2. split (20% test)
    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, 0.2, true, Some(42)).expect("split failed");

    // 3. scale (fit on train only)
    let (scaler, x_train_s) = StandardScaler::fit_transform(&x_train);
    let x_test_s = scaler.transform(&x_test);

    // 4. add bias
    let xb_train = preprocess::add_bias(&x_train_s);
    let xb_test = preprocess::add_bias(&x_test_s);

    // 5. train linear regression model
    let coef = linear::train_ols(&xb_train, &y_train).expect("OLS training failed");

    // 6. evaluate
    let y_pred_train = linear::predict(&xb_train, &coef);
    let y_pred_test = linear::predict(&xb_test, &coef);

    println!("Train RMSE: {:.4}", linear::rmse(&y_train, &y_pred_train));
    println!("Train R²:   {:.4}", linear::r2(&y_train, &y_pred_train));
    println!("Test RMSE:  {:.4}", linear::rmse(&y_test, &y_pred_test));
    println!("Test R²:    {:.4}", linear::r2(&y_test, &y_pred_test));

    println!("\nCoefficients: {:?}", coef);
}
