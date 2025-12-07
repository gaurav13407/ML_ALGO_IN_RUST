
mod data;
mod preprocess;
mod model;
mod split; // the file we added

use split::train_test_split;
use preprocess::StandardScaler;

fn main() {
    // load full X, y
    let (X, y) = data::load_csv("data.csv", 3).expect("load failed");

    // split (20% test, shuffle with seed)
    let (X_train, X_test, y_train, y_test) =
        train_test_split(&X, &y, 0.2, true, Some(42)).expect("split failed");

    println!("train: {:?}, test: {:?}", X_train.dim(), X_test.dim());

    // fit scaler on train, transform both
    let (scaler, X_train_s) = StandardScaler::fit_transform(&X_train);
    let X_test_s = scaler.transform(&X_test);

    // add bias
    let Xb_train = preprocess::add_bias(&X_train_s);
    let Xb_test = preprocess::add_bias(&X_test_s);

    // model
    let mut lr = model::LinearRegression::new(Xb_train.ncols());
    lr.fit(&Xb_train, &y_train, 2000, 0.01, true, 100);

    // evaluate on test
    let preds_test = lr.predict(&Xb_test);
    let test_loss = model::LinearRegression::mse(&preds_test, &y_test);
    println!("Test loss = {:.6}", test_loss);
}
