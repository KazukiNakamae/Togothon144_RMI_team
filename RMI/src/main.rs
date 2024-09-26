// < begin copyright > 
// Copyright Ryan Marcus 2020
// 
// See root directory of this project for license terms.
// 
// < end copyright > 

// This directive disables certain clippy lints that might raise unnecessary warnings.
// このディレクティブは、不要な警告を発する可能性のある一部のclippyリントを無効にします。
// 这个指令禁用了一些可能会产生不必要警告的clippy lint。
#![allow(clippy::needless_return)]

#[macro_use]
mod load; // Load module definitions with macro imports.
// モジュールの定義をマクロインポートと共に読み込みます。
// 使用宏导入模块定义。

use load::{load_data, DataType}; // Import functions and types for loading data.
// データの読み込みに関する関数と型をインポートします。
// 导入用于加载数据的函数和类型。
use rmi_lib::{train, train_bounded}; // Import the RMI training functions.
// RMIトレーニング関数をインポートします。
// 导入RMI训练函数。
use rmi_lib::KeyType; // Import the key type for the RMI.
// RMIのキータイプをインポートします。
// 导入RMI的键类型。
use rmi_lib::optimizer; // Import optimization utilities from the RMI library.
// RMIライブラリから最適化のユーティリティをインポートします。
// 从RMI库导入优化工具。

use json::*; // Import for handling JSON data.
// JSONデータの処理用インポート。
// 导入处理JSON数据。
use log::*; // Import logging utilities.
// ロギングユーティリティのインポート。
// 导入日志工具。
use std::f64; // Import the floating point utilities.
// 浮動小数点のユーティリティをインポートします。
// 导入浮点工具。
use std::fs::File; // File handling utilities.
// ファイル処理ユーティリティ。
// 文件处理工具。
use std::io::BufWriter; // Buffered file writing.
// バッファ付きファイル書き込み。
// 缓冲文件写入。
use std::fs; // Standard filesystem utilities.
// 標準ファイルシステムユーティリティ。
// 标准文件系统工具。
use std::path::Path; // Path utilities for working with file paths.
// ファイルパスを扱うためのパスユーティリティ。
// 用于处理文件路径的路径工具。
use rayon::prelude::*; // Import for parallel iteration and task handling.
// 並列イテレーションとタスク処理用のインポート。
// 导入并行迭代和任务处理。

use indicatif::{ProgressBar, ProgressStyle}; // For progress bar display.
// プログレスバーの表示用。
// 用于显示进度条。
use clap::{App, Arg}; // Command-line argument parsing.
// コマンドライン引数の解析。
// 解析命令行参数。

fn main() {
    env_logger::init(); // Initialize the logger.
    // ロガーを初期化します。
    // 初始化日志记录器。

    let matches = App::new("RMI Learner") // Create a new CLI application.
    // 新しいCLIアプリケーションを作成します。
    // 创建一个新的CLI应用程序。
        .version("0.2") // Specify the version.
    // バージョンを指定します。
    // 指定版本。
        .author("Youngmok Jung <tom418@kaist.ac.kr>") // Specify the author.
    // 著者を指定します。
    // 指定作者。
        .about("Learns recursive model indexes, code adapted from Ryan Marcus <ryan@ryanmarc.us>") // Brief description.
    // 簡単な説明。
    // 简要描述。
        .arg(Arg::with_name("input") // Input file argument.
    // 入力ファイルの引数。
    // 输入文件参数。
             .help("Path to input file containing data") // Input file path.
    // 入力ファイルのパス。
    // 输入文件路径。
             .index(1).required(true)) // It is required.
    // 必須項目。
    // 必填项。
        .arg(Arg::with_name("namespace") // Namespace for generated code.
    // 生成されたコードの名前空間。
    // 生成代码的命名空间。
             .help("Namespace to use in generated code")
             .index(2).required(false))
        .arg(Arg::with_name("models") // Models to use for training.
    // トレーニングに使用するモデル。
    // 用于训练的模型。
             .help("Comma-separated list of model layers, e.g. linear,linear")
             .index(3).required(false))
        .arg(Arg::with_name("branching factor") // Branching factor between model layers.
    // モデル層間の分岐係数。
    // 模型层之间的分支因子。
             .help("Branching factor between each model level")
             .index(4).required(false))
        .arg(Arg::with_name("no-code") // Option to skip code generation.
    // コード生成をスキップするオプション。
    // 跳过代码生成的选项。
             .long("no-code")
             .help("Skip code generation"))
        .arg(Arg::with_name("dump-ll-model-data") // Dump data used to train the last-level model.
    // 最終レベルのモデルをトレーニングするために使用されるデータをダンプします。
    // 导出用于训练最后一级模型的数据。
             .long("dump-ll-model-data")
             .value_name("model_index")
             .help("dump the data used to train the last-level model at index"))
        .arg(Arg::with_name("dump-ll-errors") // Dump the errors from last-level models.
    // 最終レベルのモデルのエラーをダンプします。
    // 导出最后一级模型的错误。
             .long("dump-ll-errors")
             .help("dump the errors of each last-level model to ll_errors.json"))
        .arg(Arg::with_name("stats-file") // Option to output statistics.
    // 統計情報を出力するオプション。
    // 输出统计信息的选项。
             .long("stats-file")
             .short("s")
             .value_name("file")
             .help("dump statistics about the learned model into the specified file"))
        .arg(Arg::with_name("param-grid") // Parameter grid for training.
    // トレーニングのためのパラメータグリッド。
    // 用于训练的参数网格。
             .long("param-grid")
             .value_name("file")
             .help("train the RMIs specified in the JSON file and report their errors"))
        .arg(Arg::with_name("data-path") // Directory to export parameters.
    // パラメータをエクスポートするディレクトリ。
    // 导出参数的目录。
             .long("data-path")
             .short("d")
             .value_name("dir")
             .help("exports parameters to files in this directory (default: rmi_data)"))
        .arg(Arg::with_name("no-errors") // Option to skip saving last-level errors.
    // 最終レベルのエラーを保存しないオプション。
    // 跳过保存最后一级错误的选项。
             .long("no-errors")
             .help("do not save last-level errors, and modify the RMI function signature"))
        .arg(Arg::with_name("threads") // Number of threads to use.
    // 使用するスレッドの数。
    // 使用的线程数。
             .long("threads")
             .short("t")
             .value_name("count")
             .help("number of threads to use for optimization, default = 4"))
        .arg(Arg::with_name("bounded") // Option to construct a bounded RMI.
    // 制限付きRMIを構築するオプション。
    // 构建有界RMI的选项。
             .long("bounded")
             .value_name("line_size")
             .help("construct an error-bounded RMI using the cachefix method for the given line size"))
        .arg(Arg::with_name("max-size") // Option to optimize RMI size.
    // RMIサイズを最適化するオプション。
    // 优化RMI大小的选项。
             .long("max-size")
             .value_name("BYTES")
             .help("uses the optimizer fo find an RMI with a size less than specified"))
        .arg(Arg::with_name("disable-parallel-training") // Option to disable parallel training.
    // 並列トレーニングを無効にするオプション。
    // 禁用并行训练的选项。
             .long("disable-parallel-training")
             .help("disables training multiple RMIs in parallel"))
        .arg(Arg::with_name("zero-build-time") // Option to zero out the build time.
    // ビルド時間をゼロにするオプション。
    // 清零构建时间的选项。
             .long("zero-build-time")
             .help("zero out the model build time field"))
        .arg(Arg::with_name("optimize") // Option to optimize the RMI.
    // RMIを最適化するオプション。
    // 优化RMI的选项。
             .long("optimize")
             .value_name("file")
             .help("Search for Pareto efficient RMI configurations. Specify the name of the output file."))
        .get_matches(); // Get the command-line arguments.
    // コマンドライン引数を取得します。
    // 获取命令行参数。

    // Set the default number of threads to 4 if unspecified.
    // 指定されていない場合、デフォルトでスレッド数を4に設定します。
    // 如果未指定，则将线程数默认设置为4。
    let num_threads = matches.value_of("threads")
        .map(|x| x.parse::<usize>().unwrap())
        .unwrap_or(4);
    // Configure Rayon thread pool with the specified number of threads.
    // Rayonスレッドプールを指定されたスレッド数で構成します。
    // 使用指定的线程数配置Rayon线程池。
    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
    
    let fp = matches.value_of("input").unwrap(); // Get the input file path.
    // 入力ファイルパスを取得します。
    // 获取输入文件路径。

    let data_dir = matches.value_of("data-path").unwrap_or("rmi_data"); // Get the data directory or use the default.
    // データディレクトリを取得するか、デフォルトを使用します。
    // 获取数据目录或使用默认值。

    // Ensure both `namespace` and `param-grid` are not specified at the same time.
    // `namespace`と`param-grid`が同時に指定されないようにします。
    // 确保不会同时指定`namespace`和`param-grid`。
    if matches.value_of("namespace").is_some() && matches.value_of("param-grid").is_some() {
        panic!("Can only specify one of namespace or param-grid");
    }
    
    info!("Reading {}...", fp); // Log the input file being read.
    // 読み込んでいる入力ファイルをログに出力します。
    // 记录正在读取的输入文件。

    let mut key_type = KeyType::U64; // Default key type is U64.
    // デフォルトのキータイプはU64です。
    // 默认键类型为U64。
    
    // Load the input data based on its type (u64, u32, etc.).
    // 入力データのタイプに基づいてデータをロードします（u64、u32など）。
    // 根据其类型（u64、u32等）加载输入数据。
    let (num_rows, data) = if fp.contains("uint64") {
        load_data(&fp, DataType::UINT64)
    } else if fp.contains("uint32") {
        key_type = KeyType::U32;
        load_data(&fp, DataType::UINT32)
    } 
    else if fp.contains("uint128") {
        key_type = KeyType::U128;
        load_data(&fp, DataType::UINT128)
    }
     else if fp.contains("uint512") {
        key_type = KeyType::U512;
        load_data(&fp, DataType::UINT512)
    }else if fp.contains("f64") {
        key_type = KeyType::F64;
        load_data(&fp, DataType::FLOAT64)
    } else {
        panic!("Data file must contain uint64, uint32, uint128, uint512, or f64.");
    };

    if matches.is_present("optimize") {
        // Perform RMI optimization if the `optimize` flag is set.
        // `optimize`フラグが設定されている場合、RMIの最適化を実行します。
        // 如果设置了`optimize`标志，执行RMI优化。
        let results = dynamic!(optimizer::find_pareto_efficient_configs,
                               data, 10);

        optimizer::RMIStatistics::display_table(&results); // Display optimization results.
        // 最適化結果を表示します。
        // 显示优化结果。

        let nmspc_prefix = if matches.value_of("namespace").is_some() {
            matches.value_of("namespace").unwrap()
        } else {
            let path = Path::new(fp);
            path.file_name().map(|s| s.to_str()).unwrap_or(Some("rmi")).unwrap()
        };
        
        // Create grid specifications and write them to a file.
        // グリッド仕様を作成し、ファイルに書き込みます。
        // 创建网格规范并将其写入文件。
        let grid_specs: Vec<JsonValue> = results.into_iter()
            .enumerate()
            .map(|(idx, v)| {
                let nmspc = format!("{}_{}", nmspc_prefix, idx);
                v.to_grid_spec(&nmspc)
            }).collect();

        let grid_specs_json = object!("configs" => grid_specs);
        let fp = matches.value_of("optimize").unwrap();
        let f = File::create(fp)
            .expect("Could not write optimization results file");
        let mut bw = BufWriter::new(f);
        grid_specs_json.write(&mut bw).unwrap();
        return;
    }

    // If we're not optimizing, ensure the RMI data directory exists.
// 最適化していない場合、RMIデータディレクトリが存在することを確認します。
// 如果我们不在优化，确保RMI数据目录存在。
    if !Path::new(data_dir).exists() {
        info!("The RMI data directory specified {} does not exist. Creating it.",
              data_dir);
        std::fs::create_dir_all(data_dir)
            .expect("The RMI data directory did not exist, and it could not be created.");
    }
    
    // Additional processing for parameter grid, training, and file output.
    // パラメータグリッド、トレーニング、ファイル出力に関する追加処理。
    // 参数网格、训练和文件输出的额外处理。

        // If the `param-grid` option is provided, load the parameter grid from the specified JSON file.
    // `param-grid`オプションが提供されている場合、指定されたJSONファイルからパラメータグリッドをロードします。
    // 如果提供了`param-grid`选项，则从指定的JSON文件加载参数网格。
    if let Some(param_grid) = matches.value_of("param-grid").map(|x| x.to_string()) {
        let pg = {
            let raw_json = fs::read_to_string(param_grid.clone()).unwrap(); // Read the JSON file.
    // JSONファイルを読み込みます。
    // 读取JSON文件。
            let mut as_json = json::parse(raw_json.as_str()).unwrap(); // Parse the JSON data.
    // JSONデータを解析します。
    // 解析JSON数据。
            as_json["configs"].take() // Take out the `configs` array from the JSON.
    // JSONから`configs`配列を取り出します。
    // 从JSON中取出`configs`数组。
        };

        let mut to_test = Vec::new(); // Prepare a list of configurations to test.
    // テストする構成のリストを準備します。
    // 准备要测试的配置列表。
        if let JsonValue::Array(v) = pg {
            for el in v {
                let layers = String::from(el["layers"].as_str().unwrap()); // Extract the model layers.
    // モデルレイヤーを抽出します。
    // 提取模型层。
                let branching = el["branching factor"].as_u64().unwrap(); // Extract the branching factor.
    // 分岐係数を抽出します。
    // 提取分支因子。
                let namespace = match el["namespace"].as_str() {
                    Some(s) => Some(String::from(s)), // Extract the namespace if available.
    // 名前空間が存在する場合、それを抽出します。
    // 如果有命名空间，提取它。
                    None => None
                };

                to_test.push((layers, branching, namespace)); // Add the configuration to the list.
    // 構成をリストに追加します。
    // 将配置添加到列表中。
            }

            trace!("# RMIs to train: {}", to_test.len()); // Log the number of RMIs to train.
    // トレーニングするRMIの数をログに出力します。
    // 记录要训练的RMI数量。

            let pbar = ProgressBar::new(to_test.len() as u64); // Create a progress bar.
    // プログレスバーを作成します。
    // 创建进度条。
            pbar.set_style(ProgressStyle::default_bar()
                        .template("{pos} / {len} ({msg}) {wide_bar} {eta}")); // Customize the progress bar's style.
    // プログレスバーのスタイルをカスタマイズします。
    // 自定义进度条的样式。

            let train_func =
                |(models, branch_factor, namespace): &(String, u64, Option<String>)| { // Define the training function.
    // トレーニング関数を定義します。
    // 定义训练函数。
                    trace!("Training RMI {} with branching factor {}",
                        models, *branch_factor); // Log the current training task.
    // 現在のトレーニングタスクをログに出力します。
    // 记录当前的训练任务。
                    
                    let loc_data = data.soft_copy(); // Create a soft copy of the data.
    // データのソフトコピーを作成します。
    // 创建数据的软拷贝。
                    let mut trained_model = dynamic!(train, loc_data, models, *branch_factor); // Train the RMI model.
    // RMIモデルをトレーニングします。
    // 训练RMI模型。
                    
                    let size_bs = rmi_lib::rmi_size(&trained_model); // Calculate the size of the trained model.
    // トレーニングされたモデルのサイズを計算します。
    // 计算训练模型的大小。

                    let result_obj = object! { // Store the training results as a JSON object.
    // トレーニング結果をJSONオブジェクトとして保存します。
    // 将训练结果存储为JSON对象。
                        "layers" => models.clone(),
                        "branching factor" => *branch_factor,
                        "average error" => trained_model.model_avg_error as f64,
                        "average error %" => trained_model.model_max_error as f64
                            / num_rows as f64 * 100.0,
                        "average l2 error" => trained_model.model_avg_l2_error as f64,
                        "average log2 error" => trained_model.model_avg_log2_error,
                        "max error" => trained_model.model_max_error,
                        "max error %" => trained_model.model_max_error as f64
                            / num_rows as f64 * 100.0,
                        "max log2 error" => trained_model.model_max_log2_error,
                        "size binary search" => size_bs,
                        "namespace" => namespace.clone()
                    };

                    if matches.is_present("zero-build-time") { // If the zero-build-time flag is set, zero out the build time.
    // `zero-build-time`フラグが設定されている場合、ビルド時間をゼロにします。
    // 如果设置了`zero-build-time`标志，则将构建时间清零。
                        trained_model.build_time = 0;
                    }
                    
                    if let Some(nmspc) = namespace { // If a namespace is provided, output the RMI.
    // 名前空間が指定されている場合、RMIを出力します。
    // 如果提供了命名空间，输出RMI。
                        rmi_lib::output_rmi(
                            &nmspc,
                            trained_model,
                            data_dir,
                            key_type,
                            true).unwrap();
                        
                    }
                    
                    pbar.inc(1); // Increment the progress bar.
    // プログレスバーを増加させます。
    // 增加进度条。
                    return result_obj; // Return the result object.
    // 結果オブジェクトを返します。
    // 返回结果对象。
                };

            let results: Vec<JsonValue> =
                if matches.is_present("disable-parallel-training") { // If parallel training is disabled, train sequentially.
    // 並列トレーニングが無効化されている場合、逐次的にトレーニングします。
    // 如果禁用了并行训练，则顺序训练。
                    trace!("Training models sequentially");
                    to_test.iter().map(train_func).collect()
                } else {
                    trace!("Training models in parallel"); // Otherwise, train in parallel.
    // それ以外の場合、並列でトレーニングします。
    // 否则并行训练。
                    to_test.par_iter().map(train_func).collect()
                };
            
            pbar.finish(); // Finish the progress bar when training completes.
    // トレーニングが完了したら、プログレスバーを終了します。
    // 训练完成后结束进度条。

            let f = File::create(format!("{}_results", param_grid)).expect("Could not write results file"); // Write the results to a file.
    // 結果をファイルに書き込みます。
    // 将结果写入文件。
            let mut bw = BufWriter::new(f);
            let json_results = object! { "results" => results };
            json_results.write(&mut bw).unwrap();
            
        } else {
            panic!("Configs must have an array as its value"); // Panic if the configs are not in array format.
    // 設定が配列形式でない場合、パニックします。
    // 如果配置不是数组格式，则会发生恐慌。
        }

    } else if matches.value_of("namespace").is_some() { // If a namespace is provided, train the model using that.
    // 名前空間が指定されている場合、それを使用してモデルをトレーニングします。
    // 如果提供了命名空间，使用它训练模型。
        let namespace = matches.value_of("namespace").unwrap().to_string();
        let mut trained_model = match matches.value_of("max-size") {
            None => {
                // If no max-size is specified, use the default training method.
    // max-sizeが指定されていない場合、デフォルトのトレーニング方法を使用します。
    // 如果未指定max-size，则使用默认的训练方法。
                let models = matches.value_of("models").unwrap();
                let branch_factor = matches
                    .value_of("branching factor")
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
        
                let trained_model = match matches.value_of("bounded") { // Check if bounded training is required.
    // 制限付きトレーニングが必要かどうかを確認します。
    // 检查是否需要有界训练。
                    None => dynamic!(train, data, models, branch_factor), // Perform standard training.
    // 標準的なトレーニングを実行します。
    // 执行标准训练。
                    Some(s) => {
                        let line_size = s.parse::<usize>()
                            .expect("Line size must be a positive integer."); // Ensure the line size is a positive integer.
    // ラインサイズが正の整数であることを確認します。
    // 确保行大小为正整数。
                        let d_u64 = data.into_u64()
                            .expect("Can only construct a bounded RMI on u64 data."); // Ensure the data is of type u64.
    // データがu64タイプであることを確認します。
    // 确保数据为u64类型。
                        train_bounded(&d_u64, models, branch_factor, line_size) // Perform bounded training.
    // 制限付きトレーニングを実行します。
    // 执行有界训练。
                    }
                };
                trained_model
            }
            Some(max_size_str) => {
                let max_size = max_size_str.parse::<usize>().unwrap(); // Parse the maximum size.
    // 最大サイズを解析します。
    // 解析最大大小。
                info!("Constructing RMI with size less than {}", max_size); // Log the size restriction.
    // サイズ制限をログに出力します。
    // 记录大小限制。

                let trained_model = dynamic!(rmi_lib::train_for_size, data, max_size); // Train the model with a size limit.
    // サイズ制限付きでモデルをトレーニングします。
    // 在大小限制下训练模型。
                trained_model
            }
        };
        
        let no_errors = matches.is_present("no-errors"); // Check if error saving is disabled.
    // エラー保存が無効化されているか確認します。
    // 检查是否禁用了错误保存。
        info!("Model build time: {} ms", trained_model.build_time / 1_000_000); // Log the model build time.
    // モデルのビルド時間をログに出力します。
    // 记录模型构建时间。

        info!(
            "Average model error: {} ({}%)", // Log the average error.
    // 平均エラーをログに出力します。
    // 记录平均错误。
            trained_model.model_avg_error as f64,
            trained_model.model_avg_error / num_rows as f64 * 100.0
        );
        info!(
            "Average model L2 error: {}", // Log the average L2 error.
    // 平均L2エラーをログに出力します。
    // 记录平均L2误差。
            trained_model.model_avg_l2_error
        );
        info!(
            "Average model log2 error: {}", // Log the average log2 error.
    // 平均log2エラーをログに出力します。
    // 记录平均log2误差。
            trained_model.model_avg_log2_error
        );
        info!(
            "Max model log2 error: {}", // Log the maximum log2 error.
    // 最大log2エラーをログに出力します。
    // 记录最大log2误差。
            trained_model.model_max_log2_error
        );
        info!(
            "Max model error on model {}: {} ({}%)", // Log the maximum error.
    // 最大エラーをログに出力します。
    // 记录最大误差。
            trained_model.model_max_error_idx,
            trained_model.model_max_error,
            trained_model.model_max_error as f64 / num_rows as f64 * 100.0
        );

        if true {
            
            println!("Model build time: {} ms", trained_model.build_time / 1_000_000); // Print the build time.
    // ビルド時間を出力します。
    // 输出构建时间。

            println!(
                "Average model error: {} ({}%)", // Print the average error.
    // 平均エラーを出力します。
    // 输出平均误差。
                trained_model.model_avg_error as f64,
                trained_model.model_avg_error / num_rows as f64 * 100.0
            );
            // println!("Average model L2 error: {}", trained_model.model_avg_l2_error);
            println!(
                "Average model log2 error: {}", // Print the average log2 error.
    // 平均log2エラーを出力します。
    // 输出平均log2误差。
                trained_model.model_avg_log2_error
            );
            println!(
                "Max model log2 error on model {}: {} ({}%)", // Print the maximum log2 error.
    // 最大log2エラーを出力します。
    // 输出最大log2误差。
                trained_model.model_max_error_idx,
                trained_model.model_max_log2_error,
                trained_model.model_max_error as f64 / num_rows as f64 * 100.0
            );
        }
        
        if !matches.is_present("no-code") { // If the `no-code` flag is not present, output the code.
    // `no-code`フラグがない場合、コードを出力します。
    // 如果不存在`no-code`标志，输出代码。
            if matches.is_present("zero-build-time") { // Zero out the build time if necessary.
    // 必要に応じてビルド時間をゼロにします。
    // 如果需要清零构建时间。
                trained_model.build_time = 0;
            }

            rmi_lib::output_rmi(
                &namespace,
                trained_model,
                data_dir,
                key_type,
                !no_errors).unwrap();
        } else {
            trace!("Skipping code generation due to CLI flag"); // Skip code generation if the flag is set.
    // フラグが設定されている場合、コード生成をスキップします。
    // 如果设置了标志，跳过代码生成。
        }
    } else {
        trace!("Must specify either a name space or a parameter grid."); // Log a message if neither namespace nor param-grid is specified.
    // 名前空間またはパラメータグリッドのいずれかを指定する必要があります。
    // 如果未指定命名空间或参数网格，记录消息。
    }
}
