use ndarray::{Array1,Array2};
use std::{error::Error,fs::File};
use csv::ReaderBuilder;

pub fn load_csv(path:&str,target_column:usize)->Result<(Array2<f64>,Array1<f64>),Box<dyn Error>>{
    let file=File::open(path)?;
    let mut rdr=ReaderBuilder::new().has_headers(true).flexible(true).from_reader(file);
    let mut features:Vec<Vec<f64>> =Vec::new();
    let mut targets:Vec<f64> =Vec::new();
    
    for(line_num,result) in rdr.records().enumerate(){
        let record=match result{
            Ok(r)=>r,
            Err(e)=>{
                return Err(format!("Csv Parser error at record {} :{}!",line_num+1,e).into())}};

        let mut row_vals:Vec<f64> = Vec::new();
        for field in record.iter(){
            row_vals.push(field.parse::<f64>()?);
        }
        if target_column>=row_vals.len(){
            return Err(format!("Target Column {} out of range at line {}",target_column,line_num).into());
        }
        let y_val=row_vals[target_column];
        row_vals.remove(target_column);

        targets.push(y_val);
        features.push(row_vals);
    }
    if features.is_empty(){
        return Err("Noi data rwos found in csv".into());
    }
    let n_sample=features.len();
    let n_features=features[0].len();

    let flat:Vec<f64> = features.into_iter().flatten().collect();

    let x=Array2::from_shape_vec((n_sample,n_features),flat)?;
    let y=Array1::from_vec(targets);

    Ok((x,y))
}
