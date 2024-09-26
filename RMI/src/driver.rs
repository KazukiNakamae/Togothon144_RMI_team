#![allow(clippy::needless_return)]

#[macro_use]
mod load;

use load::{ load_data, DataType };
use rmi_lib::train;
use rmi_lib::KeyType;
use rmi_lib::driver_validation;
use clap::{ App, Arg };
use log::info;
use std::path::Path;
use rmi_lib::train::ValidationResults;
use std::fs::File;
use std::io::Write;
use std::time::SystemTime;


fn main() {
    env_logger::init();

    let matches = App::new("RMI Learner")
        .version("0.1")
        .author("-")
        .about("Driver to validate RMI models")
        .arg(
            Arg::with_name("input")
                .help("Path to input file containing data")
                .index(1)
                .required(true)
        )
        .arg(
            Arg::with_name("models")
                .help("Comma-separated list of model layers, e.g. linear,linear")
                .index(2)
                .required(true)
        )
        .arg(
            Arg::with_name("branching factor")
                .help("Branching factor between each model level")
                .index(3)
                .required(false)
        )
        .get_matches();

    let fp = matches.value_of("input").unwrap();
    let models = matches.value_of("models").unwrap();
    let branching_factor = matches.value_of("branching factor").unwrap().parse::<u64>().unwrap();
    let num_threads = 4;
    let data_dir = "output";
    let mut key_type = KeyType::U64;

    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();

    println!("Reading {}...", fp);
    println!("Using {}...", models);
    println!("With brancing factor of {}...", branching_factor);
    println!("With {} threads...\n", num_threads);

    let (num_rows, data) = if fp.contains("uint64") {
        load_data(&fp, DataType::UINT64)
    } else if fp.contains("uint32") {
        key_type = KeyType::U32;
        load_data(&fp, DataType::UINT32)
    } else if fp.contains("uint128") {
        key_type = KeyType::U128;
        load_data(&fp, DataType::UINT128)
    } else if fp.contains("uint512") {
        key_type = KeyType::U512;
        load_data(&fp, DataType::UINT512)
    } else if fp.contains("f64") {
        key_type = KeyType::F64;
        load_data(&fp, DataType::FLOAT64)
    } else {
        panic!("Data file must contain uint64, uint32, uint128, uint512, or f64.");
    };

    if !Path::new(data_dir).exists() {
        info!("The output directory specified {} does not exist. Creating it.", data_dir);
        std::fs
            ::create_dir_all(data_dir)
            .expect("The output directory did not exist, and it could not be created.");
    }

    let validation_results: ValidationResults = dynamic!(driver_validation, data, models, branching_factor);
    
    let json_results = validation_results.to_json();

    let timestamp = SystemTime::now()
      .duration_since(SystemTime::UNIX_EPOCH)
      .expect("Time went backwards")
      .as_secs();

    let output_file_path = Path::new(data_dir).join(format!("validation_results_{}.json", timestamp));

    let mut file = File::create(output_file_path.clone())
      .expect("Failed to create output file");
    file.write_all(json_results.pretty(2).as_bytes())
      .expect("Failed to write to output file");

    println!("Validation results have been written to {}", output_file_path.display());
}
