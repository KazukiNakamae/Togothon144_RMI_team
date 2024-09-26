extern crate rmi_lib;

use rmi_lib::{ train::*, RMITrainingData };
use std::any::Any;
use std::env;
use std::fs::File;
use std::io::{ BufReader, BufRead, Write };

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        println!("Usage: driver <data_file> <model_spec> <branch_factor>");
        return;
    }

    let data_file = &args[1];
    let model_spec = &args[2];
    let branch_factor: u64 = args[3].parse().expect("Invalid branch factor");

    let data = load_data(data_file);

    create_directory("output");

    let output_data = driver_validation(&data, model_spec, branch_factor);

    if let Some(validation_results) = output_data {
        let res_json = format!(r#"{{"{}"}}"#, output_data);
        save_json("output/driver_validate.json", &res_json);
        println!("Output stored in 'output/driver_validate.json'.");
    } else {
        eprintln!("Failed to validate RMI models.");
    }
}

fn load_data(file_path: &str) -> RMITrainingData<u64> {
    let file = File::open(file_path).expect("Failed to open file");
    let reader = BufReader::new(file);

    let data_vec: Vec<(u64, usize)> = reader
        .lines()
        .enumerate()
        .map(|(idx, line)| {
            let key: u64 = line
                .expect("Failed to read line")
                .parse()
                .expect("Failed to parse line as u64");
            (key, idx)
        })
        .collect();

    RMITrainingData::new(Box::new(data_vec))
}

fn create_directory(name: &str) {
    let _ = std::fs::create_dir(name);
}

fn save_json(file_path: &str, content: &str) {
    let mut file = File::create(file_path).expect("Unable to create file");
    file.write_all(content.as_bytes()).expect("Unable to write data");
}
