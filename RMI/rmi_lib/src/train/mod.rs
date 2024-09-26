// < begin copyright >
// Copyright Ryan Marcus 2020
//
// See root directory of this project for license terms.
//
// < end copyright >

use crate::models::*; // Import model definitions.
// モデル定義をインポートします。
// 导入模型定义。
use crate::cache_fix::cache_fix; // Import the cache fix utility.
// キャッシュ修正ユーティリティをインポートします。
// 导入缓存修复工具。
use log::*; // Import logging macros.
// ロギングマクロをインポートします。
// 导入日志记录宏。
use std::time::SystemTime; // Import SystemTime to measure time.
// 時間を測定するためにSystemTimeをインポートします。
// 导入SystemTime用于测量时间。

use rug::{ // Import the `rug` library for high-precision math operations.
    // 高精度数学操作のために `rug` ライブラリをインポートします。
    // 导入 `rug` 库进行高精度数学操作。
    float::{ self, FreeCache, Round },
    ops::{ AddAssignRound, AssignRound, MulAssignRound },
    Float,
    Assign,
    Integer,
};

use std::any::Any;
use json::JsonValue;

mod two_layer; // Import the two-layer model module.
// 2層モデルモジュールをインポートします。
// 导入双层模型模块。
mod multi_layer; // Uncomment if using multi-layer model module.
// マルチレイヤーモデルモジュールを使用する場合、コメントを解除します。
// 如果使用多层模型模块，请取消注释。
mod lower_bound_correction; // Import the lower-bound correction module.
// 下限補正モジュールをインポートします。
// 导入下界修正模块。

// Definition of the TrainedRMI structure, which stores the trained model information.
// トレーニングされたモデル情報を格納するTrainedRMI構造体の定義。
// TrainedRMI结构体定义，用于存储训练好的模型信息。
pub struct TrainedRMI {
    pub num_rmi_rows: usize, // Number of rows in the RMI.
    // RMI内の行数。
    // RMI中的行数。
    pub num_data_rows: usize, // Number of rows in the data.
    // データ内の行数。
    // 数据中的行数。
    pub model_avg_error: f64, // Average error of the model.
    // モデルの平均誤差。
    // 模型的平均误差。
    pub model_avg_l2_error: f64, // Average L2 error of the model.
    // モデルのL2誤差。
    // 模型的L2误差。
    pub model_avg_log2_error: f64, // Average log2 error.
    // 平均log2誤差。
    // 平均log2误差。
    pub model_max_error: u64, // Maximum error encountered.
    // 遭遇した最大誤差。
    // 遇到的最大误差。
    pub model_max_error_idx: usize, // Index of the data point with the maximum error.
    // 最大誤差を持つデータポイントのインデックス。
    // 具有最大误差的数据点的索引。
    pub model_max_log2_error: f64, // Maximum log2 error encountered.
    // 遭遇した最大log2誤差。
    // 遇到的最大log2误差。
    pub last_layer_max_l1s: Vec<u64>, // Maximum L1 errors for the last layer models.
    // 最終レイヤーモデルの最大L1誤差。
    // 最后一层模型的最大L1误差。
    pub third_layer_max_l1s: Vec<u64>, // Maximum L1 errors for the third layer models (if present).
    // 第三レイヤーモデルの最大L1誤差（存在する場合）。
    // 第三层模型的最大L1误差（如果存在）。
    pub rmi: Vec<Vec<Box<dyn Model>>>, // The RMI itself, consisting of multiple layers of models.
    // 複数のレイヤーのモデルで構成されるRMI自体。
    // RMI本身，由多个模型层组成。
    pub models: String, // A string representing the model types used.
    // 使用されたモデルタイプを表す文字列。
    // 表示使用的模型类型的字符串。
    pub branching_factor: u64, // Branching factor for the RMI.
    // RMIの分岐係数。
    // RMI的分支因子。
    pub cache_fix: Option<(usize, Vec<(u64, usize)>)>, // Optional cache fix applied to the model.
    // モデルに適用されたオプションのキャッシュ修正。
    // 应用于模型的可选缓存修复。
    pub build_time: u128, // Time taken to build the RMI.
    // RMIの構築にかかった時間。
    // 构建RMI所花费的时间。
}

impl TrainedRMI {
    pub fn to_json(&self) -> JsonValue {
        json::object! {
            "num_rmi_rows" => self.num_rmi_rows,
            "num_data_rows" => self.num_data_rows,
            "model_avg_error" => self.model_avg_error,
            "model_avg_l2_error" => self.model_avg_l2_error,
            "model_avg_log2_error" => self.model_avg_log2_error,
            "model_max_error" => self.model_max_error,
            "model_max_error_idx" => self.model_max_error_idx,
            "model_max_log2_error" => self.model_max_log2_error,
            "last_layer_max_l1s" => JsonValue::from(self.last_layer_max_l1s.iter().map(|&x| x.to_string()).collect::<Vec<String>>()),
            "third_layer_max_l1s" => JsonValue::from(self.third_layer_max_l1s.iter().map(|&x| x.to_string()).collect::<Vec<String>>()),
            "models" => self.models.clone(),
            "branching_factor" => self.branching_factor.to_string(),
            "build_time" => self.build_time.to_string()
            // Note: We're omitting 'rmi' and 'cache_fix' fields as they might be complex to serialize
        }
    }
}

// Train a "big" model based on the specified type and data.
// 指定されたタイプとデータに基づいて「ビッグ」モデルをトレーニングします。
// 根据指定的类型和数据训练一个“大”模型。
fn train_model_big<T: TrainingKey>(model_type: &str, data: &RMITrainingData<T>) -> Box<dyn Model> {
    let model: Box<dyn Model> = match model_type {
        "linear" => Box::new(LinearModelBig::new(data)), // Train a big linear model.
        // 大きな線形モデルをトレーニングします。
        // 训练一个大线性模型。
        _ => panic!("Unknown model type: {}", model_type), // Handle unknown model types.
        // 未知のモデルタイプを処理します。
        // 处理未知的模型类型。
    };

    return model;
}

// Train a model based on the specified type and data.
// 指定されたタイプとデータに基づいてモデルをトレーニングします。
// 根据指定的类型和数据训练一个模型。
fn train_model<T: TrainingKey>(model_type: &str, data: &RMITrainingData<T>) -> Box<dyn Model> {
    let model: Box<dyn Model> = match model_type {
        "linear_big" => Box::new(LinearModelBig::new(data)), // Train a large linear model.
        // 大きな線形モデルをトレーニングします。
        // 训练一个大线性模型。
        "linear" => Box::new(LinearModel::new(data)), // Train a standard linear model.
        // 標準の線形モデルをトレーニングします。
        // 训练一个标准线性模型。
        "pwl" => Box::new(PiecewiselinearModel::new(data, 28)), // Train a piecewise linear model with 28 segments.
        // 28セグメントの区分線形モデルをトレーニングします。
        // 训练一个有28段的分段线性模型。
        "pwl30" => Box::new(PiecewiselinearModel::new(data, 30)), // Train a piecewise linear model with 30 segments.
        // 30セグメントの区分線形モデルをトレーニングします。
        // 训练一个有30段的分段线性模型。
        // Additional models are defined here...
        // 他のモデルもここで定義されます。
        // 这里还定义了其他模型。
        _ => panic!("Unknown model type: {}", model_type), // Handle unknown model types.
        // 未知のモデルタイプを処理します。
        // 处理未知的模型类型。
    };

    return model;
}

// Validate the model specification to ensure the correct order of models in layers.
// レイヤー内のモデルの順序が正しいことを確認するためにモデル仕様を検証します。
// 验证模型规格以确保层中的模型顺序正确。
fn validate(model_spec: &[String]) {
    let num_layers = model_spec.len(); // Number of layers specified.
    // 指定されたレイヤー数。
    // 指定的层数。
    let empty_container: RMITrainingData<u64> = RMITrainingData::empty(); // Create an empty RMITrainingData container.
    // 空のRMITrainingDataコンテナを作成します。
    // 创建一个空的RMITrainingData容器。

    for (idx, model) in model_spec.iter().enumerate() {
        let restriction = train_model(model, &empty_container).restriction(); // Get the model's restriction (if any).
        // モデルの制限（ある場合）を取得します。
        // 获取模型的限制（如果有）。

        match restriction {
            ModelRestriction::None => {} // No restrictions for this model.
            // このモデルには制限がありません。
            // 该模型没有限制。
            ModelRestriction::MustBeTop => {
                // Ensure this model is used at the top layer if required.
                // このモデルがトップレイヤーで使用されることを確認します。
                // 确保该模型在顶层使用。
                assert_eq!(idx, 0, "if used, model type {} must be the root model", model);
            }
            ModelRestriction::MustBeBottom => {
                // Ensure this model is used at the bottom layer if required.
                // このモデルがボトムレイヤーで使用されることを確認します。
                // 确保该模型在底层使用。
                assert_eq!(
                    idx,
                    num_layers - 1,
                    "if used, model type {} must be the bottommost model",
                    model
                );
            }
        }
    }
}

// Train an RMI based on the given model specification and branching factor.
// 指定されたモデル仕様と分岐係数に基づいてRMIをトレーニングします。
// 根据给定的模型规格和分支因子训练RMI。
pub fn train<T: TrainingKey>(
    data: &RMITrainingData<T>,
    model_spec: &str,
    branch_factor: u64
) -> TrainedRMI {
    let start_time = SystemTime::now(); // Start timing the training process.
    // トレーニングプロセスのタイミングを開始します。
    // 开始计时训练过程。
    let (model_list, last_model): (Vec<String>, String) = {
        // Split the model specification into layers and extract the last model.
        // モデル仕様をレイヤーに分割し、最後のモデルを抽出します。
        // 将模型规格分成层并提取最后一个模型。
        let mut all_models: Vec<String> = model_spec.split(',').map(String::from).collect();
        validate(&all_models); // Validate the model specification.
        // モデル仕様を検証します。
        // 验证模型规格。
        let last = all_models.pop().unwrap();
        (all_models, last)
    };

    if model_list.len() == 1 {
        // If the specification contains only one layer, train a two-layer RMI.
        // 仕様に1つのレイヤーしか含まれていない場合、2層RMIをトレーニングします。
        // 如果规格只包含一层，则训练一个双层RMI。
        let mut res = two_layer::train_two_layer(
            &mut data.soft_copy(),
            &model_list[0],
            &last_model,
            branch_factor
        );
        let build_time = SystemTime::now()
            .duration_since(start_time)
            .map(|d| d.as_nanos())
            .unwrap_or(std::u128::MAX); // Calculate the build time.
        // ビルド時間を計算します。
        // 计算构建时间。
        res.build_time = build_time;

        return res;
    }

    if model_list.len() == 2 {
        // If the specification contains two layers, train a partial three-layer RMI.
        // 仕様に2つのレイヤーが含まれている場合、部分的な3層RMIをトレーニングします。
        // 如果规格包含两层，则训练一个部分三层RMI。
        let mut res = two_layer::train_partial_three_layer(
            &mut data.soft_copy(),
            &model_list[0],
            &model_list[1],
            &last_model,
            branch_factor
        );
        let build_time = SystemTime::now()
            .duration_since(start_time)
            .map(|d| d.as_nanos())
            .unwrap_or(std::u128::MAX); // Calculate the build time.
        // ビルド時間を計算します。
        // 计算构建时间。
        res.build_time = build_time;

        return res;
    }

    // If more than two layers are specified, training should be expanded to support multiple layers.
    // 2つ以上のレイヤーが指定されている場合、トレーニングは複数のレイヤーをサポートするように拡張されるべきです。
    // 如果指定了两层以上，则训练应扩展以支持多层。
    panic!(); // TODO: Implement multi-layer RMI training.
    // マルチレイヤーRMIトレーニングを実装する必要があります。
    // 需要实现多层RMI训练。
}

// Train an RMI with a size constraint.
// サイズ制約を持つRMIをトレーニングします。
// 训练具有大小约束的RMI。
pub fn train_for_size<T: TrainingKey>(data: &RMITrainingData<T>, max_size: usize) -> TrainedRMI {
    let start_time = SystemTime::now(); // Start timing the training process.
    // トレーニングプロセスのタイミングを開始します。
    // 开始计时训练过程。
    let pareto = crate::find_pareto_efficient_configs(data, 1000); // Find the Pareto-efficient configurations.
    // パレート効率の良い構成を見つけます。
    // 找到帕累托效率配置。

    // Select the first configuration that meets the size constraint.
    // サイズ制約を満たす最初の構成を選択します。
    // 选择满足大小约束的第一个配置。
    let config = pareto
        .into_iter()
        .filter(|x| x.size < (max_size as u64))
        .next()
        .expect(format!("Could not find any configurations smaller than {}", max_size).as_str());

    let models = config.models; // Extract the model specification.
    // モデル仕様を抽出します。
    // 提取模型规格。
    let bf = config.branching_factor; // Extract the branching factor.
    // 分岐係数を抽出します。
    // 提取分支因子。

    info!(
        "Found RMI config {} {} with size {} and average log2 {}",
        models,
        bf,
        config.size,
        config.average_log2_error
    ); // Log the configuration.
    // 構成をログに記録します。
    // 记录配置。

    let mut res = train(data, models.as_str(), bf); // Train the RMI with the selected configuration.
    // 選択された構成でRMIをトレーニングします。
    // 使用所选配置训练RMI。

    let build_time = SystemTime::now()
        .duration_since(start_time)
        .map(|d| d.as_nanos())
        .unwrap_or(std::u128::MAX); // Calculate the build time.
    // ビルド時間を計算します。
    // 计算构建时间。
    res.build_time = build_time;
    return res;
}

// Train an error-bounded RMI using cache fixing.
// キャッシュ修正を使用してエラーが制限されたRMIをトレーニングします。
// 使用缓存修复训练带有误差界限的RMI。
pub fn train_bounded(
    data: &RMITrainingData<u64>,
    model_spec: &str,
    branch_factor: u64,
    line_size: usize
) -> TrainedRMI {
    let start_time = SystemTime::now(); // Start timing the training process.
    // トレーニングプロセスのタイミングを開始します。
    // 开始计时训练过程。

    // Transform the data into error-bounded spline points using cache fixing.
    // キャッシュ修正を使用してデータをエラー制限付きスプラインポイントに変換します。
    // 使用缓存修复将数据转换为具有误差界限的样条点。
    let spline = cache_fix(data, line_size);
    std::mem::drop(data); // Release the original data from memory.
    // 元のデータをメモリから解放します。
    // 从内存中释放原始数据。

    // Reindex the spline points so they can be used to build an RMI.
    // スプラインポイントのインデックスを再作成して、RMIを構築できるようにします。
    // 重新索引样条点以便用于构建RMI。
    let reindexed_splines: Vec<(u64, usize)> = spline
        .iter()
        .enumerate()
        .map(|(idx, (key, _old_offset))| (*key, idx))
        .collect();

    // Construct new training data from the spline points.
    // スプラインポイントから新しいトレーニングデータを構築します。
    // 从样条点构造新的训练数据。
    let mut new_data = RMITrainingData::new(Box::new(reindexed_splines));

    let mut res = crate::train(&mut new_data, model_spec, branch_factor); // Train the RMI on the spline data.
    // スプラインデータに基づいてRMIをトレーニングします。
    // 在样条数据上训练RMI。
    res.cache_fix = Some((line_size, spline)); // Store the cache fix information.
    // キャッシュ修正情報を保存します。
    // 存储缓存修复信息。
    res.num_data_rows = data.len(); // Store the number of data rows.
    // データ行数を保存します。
    // 存储数据行数。

    let build_time = SystemTime::now()
        .duration_since(start_time)
        .map(|d| d.as_nanos())
        .unwrap_or(std::u128::MAX); // Calculate the build time.
    // ビルド時間を計算します。
    // 计算构建时间。
    res.build_time = build_time;
    return res;
}

// for driver to validate RMI models
pub struct ValidationResults {
    pub multi_layer: TrainedRMI,
    pub two_layer: Option<TrainedRMI>,
    pub three_layer: Option<TrainedRMI>,
    pub naive_three_layer: Option<TrainedRMI>,
    pub partial_three_layer: Option<TrainedRMI>,
}

impl ValidationResults {
    pub fn to_json(&self) -> JsonValue {
        json::object! {
            "multi_layer" => self.multi_layer.to_json(),
            "two_layer" => self.two_layer.as_ref().map(|rmi| rmi.to_json()),
            "three_layer" => self.three_layer.as_ref().map(|rmi| rmi.to_json()),
            "naive_three_layer" => self.naive_three_layer.as_ref().map(|rmi| rmi.to_json()),
            "partial_three_layer" => self.partial_three_layer.as_ref().map(|rmi| rmi.to_json())
        }
    }
}

pub fn driver_validation<T: TrainingKey>(
    data: &RMITrainingData<T>,
    model_spec: &str,
    branch_factor: u64
) -> ValidationResults {
    let (model_list, last_model): (Vec<String>, String) = {
        let mut all_models: Vec<String> = model_spec.split(',').map(String::from).collect();
        validate(&all_models);
        let last = all_models.pop().unwrap();
        (all_models, last)
    };

    // TRAIN MULTI LAYER
    let start_time = SystemTime::now();
    let mut train_multi_layer_res = multi_layer::train_multi_layer(
        &mut data.soft_copy(),
        &model_list,
        last_model.clone(),
        branch_factor
    );
    let train_multi_layer_build_time = SystemTime::now()
        .duration_since(start_time)
        .map(|d| d.as_nanos())
        .unwrap_or(std::u128::MAX);
    train_multi_layer_res.build_time = train_multi_layer_build_time;

    // TRAIN TWO LAYER
    let train_two_layer_res = if model_list.len() >= 1 {
        let start_time = SystemTime::now();
        let mut res = two_layer::train_two_layer(
            &mut data.soft_copy(),
            &model_list[0],
            &last_model,
            branch_factor
        );
        let train_two_layer_build_time = SystemTime::now()
            .duration_since(start_time)
            .map(|d| d.as_nanos())
            .unwrap_or(std::u128::MAX);
        res.build_time = train_two_layer_build_time;
        Some(res)
    } else {
        println!("Warning: Not enough layers specified for two-layer model");
        None
    };

    // TRAIN THREE LAYER, NAIVE THREE LAYER, AND PARTIAL THREE LAYER
    let (train_three_layer_res, train_naive_three_layer_res, train_partial_three_layer_res) = 
    if model_list.len() >= 2 {
        let start_time = SystemTime::now();
        let mut three_layer = two_layer::train_three_layer(
            &mut data.soft_copy(),
            &model_list[0],
            &model_list[1],
            &last_model,
            branch_factor
        );
        let train_three_layer_build_time = SystemTime::now()
            .duration_since(start_time)
            .map(|d| d.as_nanos())
            .unwrap_or(std::u128::MAX);
        three_layer.build_time = train_three_layer_build_time;

        let start_time = SystemTime::now();
        let mut naive_three_layer = two_layer::train_naive_three_layer(
            &mut data.soft_copy(),
            &model_list[0],
            &model_list[1],
            &last_model,
            branch_factor
        );
        let train_naive_three_layer_build_time = SystemTime::now()
            .duration_since(start_time)
            .map(|d| d.as_nanos())
            .unwrap_or(std::u128::MAX);
        naive_three_layer.build_time = train_naive_three_layer_build_time;

        let start_time = SystemTime::now();
        let mut partial_three_layer = two_layer::train_partial_three_layer(
            &mut data.soft_copy(),
            &model_list[0],
            &model_list[1],
            &last_model,
            branch_factor
        );
        let train_partial_three_layer_build_time = SystemTime::now()
            .duration_since(start_time)
            .map(|d| d.as_nanos())
            .unwrap_or(std::u128::MAX);
        partial_three_layer.build_time = train_partial_three_layer_build_time;

        (Some(three_layer), Some(naive_three_layer), Some(partial_three_layer))
    } else {
        println!("Warning: Not enough layers specified for three-layer models");
        (None, None, None)
    };

    ValidationResults {
        multi_layer: train_multi_layer_res,
        two_layer: train_two_layer_res,
        three_layer: train_three_layer_res,
        naive_three_layer: train_naive_three_layer_res,
        partial_three_layer: train_partial_three_layer_res,
    }
}
