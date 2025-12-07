mod data;
mod preprocess;
mod split;
mod logistic;

use preprocess::StandardScaler;
use split::train_test_split;

fn main() {
    // 1. load data
    let (X, y) = data::load_csv("data.csv", 3).expect("load failed");

    // 2. split (20% test)
    let (X_train, X_test, y_train, y_test) =
        train_test_split(&X, &y, 0.2, true, Some(42)).expect("split failed");

    // 3. scale (fit on train only)
    let (scaler, X_train_s) = StandardScaler::fit_transform(&X_train);
    let X_test_s = scaler.transform(&X_test);

    // 4. add bias
    let Xb_train = preprocess::add_bias(&X_train_s);
    let Xb_test = preprocess::add_bias(&X_test_s);

    // 5. create and train logistic model
    let mut clf = logistic::LogisticRegression::new(Xb_train.ncols());
    clf.fit(&Xb_train, &y_train, 2000, 0.1, true, 100);

    // 6. evaluate
    let probs = clf.predict_proba(&Xb_test);
    let preds = clf.predict(&Xb_test);
    let loss = logistic::LogisticRegression::log_loss(&probs, &y_test);

    println!("Test log_loss = {:.6}", loss);
    let acc = preds
        .iter()
        .zip(y_test.iter())
        .filter(|(p, yt)| (*p - *yt).abs() < 1e-9)
        .count() as f64
        / (y_test.len() as f64);
    println!("Test accuracy = {:.4}", acc);

    println!("Weights: {:?}", clf.w);
}

