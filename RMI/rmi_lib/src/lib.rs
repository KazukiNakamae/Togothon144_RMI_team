mod codegen;
mod models;
pub mod train;
mod cache_fix;

pub mod optimizer;
pub use models::{RMITrainingData, RMITrainingDataIteratorProvider, ModelInput};
pub use models::KeyType;
pub use models::U512;
pub use optimizer::find_pareto_efficient_configs;
pub use train::{train, train_for_size, train_bounded, driver_validation };
pub use codegen::rmi_size;
pub use codegen::output_rmi;