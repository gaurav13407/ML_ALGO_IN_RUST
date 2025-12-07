use ndarray::{Array1,Array2};
use std::fmt;

pub struct LinearRegression{
    pub w:Array1<f64>,
}

impl fmt::Debug for LinearRegression{
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        write!(f,"LinearRegression{{w:{:?}}}",self.w)
    }
}

impl LinearRegression{
    pub fn new(n_features_with_bias:usize)->Self{
        let w=Array1::<f64>::zeros(n_features_with_bias);
        Self{w}
    }
    pub fn predict(&self,X:&Array2<f64>)->Array1<f64>{
        X.dot(&self.w)
    }

    // Mean Squared Error
    pub fn mse(pred: &Array1<f64>,y:&Array1<f64>)->f64{
        let diff=pred-y;

        diff.mapv(|v|v*v).mean().unwrap_or(0.0)
    }
 /// Full-batch gradient descent
    /// - X: (n_samples, n_features_with_bias) — already normalized + bias column
    /// - y: (n_samples,)
    /// - epochs: number of passes over the data
    /// - lr: learning rate (e.g., 0.01)
    /// - verbose: if true prints loss every `print_every` epochs
    
    pub fn  fit(&mut self,X:&Array2<f64>,y:&Array1<f64>,epochs:usize,lr:f64,verbose:bool,print_every:usize){
        let n=X.nrows() as f64;

        for epoch in 0..epochs{
            let preds=self.predict(X);

            let residual=&preds-y;

            let grad=(2.0/n)*X.t().dot(&residual);

            self.w=&self.w-&(grad*lr);

            if verbose && (epochs % print_every==0 || epoch+1==epochs){
                let loss=Self::mse(&preds,y);
                println!("epochs {:>4}/{:<4} loss ={:.6}",epoch+1,epochs,loss);
            }
                
        }
    }
}
