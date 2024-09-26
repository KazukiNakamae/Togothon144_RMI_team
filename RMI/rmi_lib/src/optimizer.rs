use crate::models::*;
use crate::train;
use crate::codegen;
use log::*;
use json::*;
use indicatif::{ProgressBar};
use rayon::prelude::*;
use std::collections::BTreeSet;
use tabular::{Table, row};

// Use constants to define different model layers for the optimizer to choose from.
// 最適化で選択される異なるモデル層を定義するために定数を使用します。
// 使用常量定义优化器可以选择的不同模型层。
fn top_only_layers() -> Vec<&'static str> {
    // Return the top-level layers based on the optimizer profile.
    // 最適化プロファイルに基づいてトップレベルの層を返します。
    // 根据优化器配置文件返回顶层模型。
    return match std::env::var_os("RMI_OPTIMIZER_PROFILE") {
        None => vec!["radix", "radix18", "radix22", "robust_linear"], // Default top-level layers.
        // デフォルトのトップレベルの層。
        // 默认顶层模型。
        Some(x) => {
            // Modify the top-level layers based on different profiles.
            // 異なるプロファイルに基づいてトップレベルの層を変更します。
            // 根据不同的配置修改顶层模型。
            match x.to_str().unwrap() {
                "fast" => vec!["robust_linear"], // Fast profile uses fewer models.
                // 高速プロファイルは少数のモデルを使用します。
                // 快速配置使用较少的模型。
                "memory" => vec!["radix", "radix18", "radix22", "robust_linear"], // Memory profile uses more.
                // メモリプロファイルはより多くのモデルを使用します。
                // 内存配置使用更多的模型。
                "disk" => vec!["radix", "radix18", "radix22", "robust_linear", "normal", "lognormal", "loglinear"],
                // Disk profile adds additional models.
                // ディスクプロファイルでは追加のモデルを追加します。
                // 磁盘配置增加了额外的模型。
                _ => panic!("Invalid optimizer profile {}", x.to_str().unwrap()) // Handle invalid profiles.
                // 無効なプロファイルの処理。
                // 处理无效的配置。
            }
        }
    };
}

fn anywhere_layers() -> Vec<&'static str> {
    // Return models that can appear anywhere in the RMI.
    // RMIの任意の場所に現れるモデルを返します。
    // 返回可以出现在RMI中的任何模型。
    return match std::env::var_os("RMI_OPTIMIZER_PROFILE") {
        None => vec!["linear", "cubic", "linear_spline"], // Default layers.
        // デフォルトの層。
        // 默认模型层。
        Some(x) => {
            // Modify based on the profile.
            // プロファイルに基づいて変更します。
            // 根据配置修改。
            match x.to_str().unwrap() {
                "fast" => vec!["linear", "cubic"],
                "memory" | "disk" => vec!["linear", "cubic", "linear_spline"],
                _ => panic!("Invalid optimizer profile {}", x.to_str().unwrap())
            }
        }
    };
}

fn get_branching_factors() -> Vec<u64> {
    // Return the branching factors for the RMI.
    // RMIの分岐係数を返します。
    // 返回RMI的分支因子。
    let range = match std::env::var_os("RMI_OPTIMIZER_PROFILE") {
        None => (6..25).step_by(1),
        // For each profile, adjust the range and step size of the branching factor.
        // 各プロファイルに対して、分岐係数の範囲とステップサイズを調整します。
        // 对每个配置调整分支因子的范围和步长。
        Some(x) => {
            match x.to_str().unwrap() {
                "fast" => (6..25).step_by(2),
                "memory" => (6..25).step_by(1),
                "disk" => (6..28).step_by(1),
                _ => panic!("Invalid optimizer profile {}", x.to_str().unwrap())
            }
        }
    };

    // Generate and return the branching factors as powers of 2.
    // 2のべき乗として分岐係数を生成して返します。
    // 以2的幂生成并返回分支因子。
    return range.map(|i| (2 as u64).pow(i)).collect();
}

fn pareto_front(results: &[RMIStatistics]) -> Vec<RMIStatistics> {
    // This function finds the Pareto frontier of the given set of RMIs.
    // この関数は、与えられたRMIセットのパレート最適曲面を見つけます。
    // 该函数找到给定RMI集的帕累托前沿。
    let mut on_front: Vec<RMIStatistics> = Vec::new();

    for result in results.iter() {
        // If `result` is dominated by any other result, it's not on the Pareto front.
        // `result`が他の結果に支配されている場合、それはパレート最適曲面にはありません。
        // 如果`result`被任何其他结果支配，则它不在帕累托前沿上。
        if results.iter().any(|v| result.dominated_by(v)) {
            continue;
        }

        // Add the result to the Pareto front if it is not dominated.
        // 支配されていない場合は、結果をパレート最適曲面に追加します。
        // 如果没有被支配，则将结果添加到帕累托前沿。
        on_front.push(result.clone());
    }

    return on_front;
}

fn narrow_front(results: &[RMIStatistics], desired_size: usize) -> Vec<RMIStatistics> {
    // Reduce the size of the Pareto front to the desired size.
    // パレート最適曲面のサイズを希望のサイズに縮小します。
    // 将帕累托前沿的大小缩小到期望的大小。
    assert!(desired_size >= 2);
    if results.len() <= desired_size {
        return results.to_vec();
    }

    let mut tmp = results.to_vec();
    tmp.sort_by(|a, b| a.size.partial_cmp(&b.size).unwrap()); // Sort by size.
// サイズでソートします。
// 按大小排序。

    let best_mod = tmp.remove(0); // Keep the smallest model.
// 最小のモデルを保持します。
// 保留最小的模型。
    while tmp.len() > desired_size - 1 {
        // Find the two closest models in size and remove the less accurate one.
        // サイズが最も近い2つのモデルを見つけ、精度の低い方を削除します。
        // 找到大小最接近的两个模型，并删除精度较低的那个。
        let smallest_gap =
            (0..tmp.len()-1).zip(1..tmp.len())
            .map(|(idx1, idx2)| (idx1, idx2,
                                 (tmp[idx2].size as f64) / (tmp[idx1].size as f64)))
            .min_by(|(_, _, v1), (_, _, v2)| v1.partial_cmp(v2).unwrap()).unwrap();

        let err1 = tmp[smallest_gap.0].average_log2_error;
        let err2 = tmp[smallest_gap.1].average_log2_error;
        if err1 > err2 {
            tmp.remove(smallest_gap.0);
        } else {
            tmp.remove(smallest_gap.1);
        }
    }
    tmp.insert(0, best_mod); // Reinsert the best model.
// 最良のモデルを再挿入します。
// 重新插入最佳模型。

    return tmp;
}

fn first_phase_configs() -> Vec<(String, u64)> {
    // Generate initial configurations for the first phase of optimization.
    // 最適化の第一フェーズ用の初期構成を生成します。
    // 生成优化第一阶段的初始配置。
    let mut results = Vec::new();
    let mut all_top_models = Vec::new();
    all_top_models.extend(top_only_layers()); // Include all top-only layers.
    // トップのみの層を含めます。
    // 包含所有仅限顶层的模型层。
    all_top_models.extend(anywhere_layers()); // Include layers that can appear anywhere.
    // 任意の場所に現れる層も含めます。
    // 包含可以出现在任何地方的模型层。
    
    for top_model in all_top_models {
        for bottom_model in anywhere_layers() {
            for branching_factor in get_branching_factors().iter().step_by(5) {
                results.push((format!("{},{}", top_model, bottom_model), *branching_factor));
                // Add each configuration of top, bottom models, and branching factors.
                // トップ、ボトムのモデル、および分岐係数の各構成を追加します。
                // 添加顶层和底层模型及分支因子的每个配置。
            }
        }
    }

    return results;
}

fn second_phase_configs(first_phase: &[RMIStatistics]) -> Vec<(String, u64)> {
    // Generate second phase configurations based on the first phase results.
    // 第一フェーズの結果に基づいて第二フェーズの構成を生成します。
    // 根据第一阶段的结果生成第二阶段的配置。
    let qualifying_model_configs = {
        let on_front = pareto_front(first_phase); // Get the Pareto front from the first phase.
        // 第一フェーズからパレート最適曲面を取得します。
        // 从第一阶段获取帕累托前沿。
        let mut qualifying = BTreeSet::new(); // Use a set to avoid duplicates.
        // 重複を避けるためにセットを使用します。
        // 使用集合避免重复。
        for result in on_front {
            qualifying.insert(result.models.clone());
        }
        qualifying
    };

    info!("Qualifying model types for phase 2: {:?}", qualifying_model_configs);
    let mut results = Vec::new();

    for model in qualifying_model_configs.iter() {
        for branching_factor in get_branching_factors() {
            if first_phase.iter().any(|v| v.has_config(&model, branching_factor)) {
                continue; // Skip configurations already used in the first phase.
                // 第一フェーズですでに使用された構成をスキップします。
                // 跳过第一阶段已经使用的配置。
            }

            results.push((model.clone(), branching_factor)); // Add the new configurations.
            // 新しい構成を追加します。
            // 添加新配置。
        }
    }
    
    return results;
}

#[derive(Clone, Debug)]
pub struct RMIStatistics {
    pub models: String,
    pub branching_factor: u64,
    pub average_log2_error: f64,
    pub max_log2_error: f64,
    pub size: u64
}

impl RMIStatistics {
    fn from_trained(rmi: &train::TrainedRMI) -> RMIStatistics {
        // Convert a trained RMI into an RMIStatistics object.
        // トレーニングされたRMIをRMIStatisticsオブジェクトに変換します。
        // 将训练的RMI转换为RMIStatistics对象。
        return RMIStatistics {
            average_log2_error: rmi.model_avg_log2_error,
            max_log2_error: rmi.model_max_log2_error,
            size: codegen::rmi_size(&rmi),
            models: rmi.models.clone(),
            branching_factor: rmi.branching_factor
        };
    }

    fn dominated_by(&self, other: &RMIStatistics) -> bool {
        // Check if this model is dominated by another model.
        // このモデルが他のモデルに支配されているかどうかを確認します。
        // 检查该模型是否被另一个模型支配。
        if self.size < other.size { return false; }
        if self.average_log2_error < other.average_log2_error { return false; }

        if self.size == other.size && self.average_log2_error <= other.average_log2_error {
            return false;
        }

        let log2_diff = (self.average_log2_error - other.average_log2_error).abs();
        if self.size <= other.size && log2_diff < std::f64::EPSILON {
            return false;
        }

        return true;
    }

    fn has_config(&self, models: &str, branching_factor: u64) -> bool {
        // Check if this model has the specified configuration.
        // このモデルが指定された構成を持っているかどうかを確認します。
        // 检查该模型是否具有指定的配置。
        return self.models == models && self.branching_factor == branching_factor;
    }

    pub fn display_table(itms: &[RMIStatistics]) {
        // Display a table of RMI statistics.
        // RMI統計の表を表示します。
        // 显示RMI统计的表格。
        let mut table = Table::new("{:<} {:>} {:>} {:>} {:>}");
        table.add_row(row!("Models", "Branch", "   AvgLg2",
                           "   MaxLg2", "   Size (b)"));
        for itm in itms {
            table.add_row(row!(itm.models.clone(),
                               format!("{:10}", itm.branching_factor),
                               format!("     {:2.5}", itm.average_log2_error),
                               format!("     {:2.5}", itm.max_log2_error),
                               format!("     {}", itm.size)));
        }

        print!("{}", table); // Print the table to the console.
        // テーブルをコンソールに出力します。
        // 将表格打印到控制台。
    }
    
    pub fn to_grid_spec(&self, namespace: &str) -> JsonValue {
        // Convert the RMI statistics into a JSON object for the parameter grid.
        // RMI統計をパラメータグリッド用のJSONオブジェクトに変換します。
        // 将RMI统计转换为参数网格的JSON对象。
        return object!(
            "layers" => self.models.clone(),
            "branching factor" => self.branching_factor,
            "namespace" => namespace,
            "size" => self.size,
            "average log2 error" => self.average_log2_error,
            "binary" => true
        );
    }
}

fn measure_rmis<T: TrainingKey>(data: &RMITrainingData<T>,
                configs: &[(String, u64)]) -> Vec<RMIStatistics> {
    // Measure the performance of different RMI configurations.
    // 異なるRMI構成のパフォーマンスを測定します。
    // 测量不同RMI配置的性能。
    let pbar = ProgressBar::new(configs.len() as u64);
    
   configs.par_iter()
        .map(|(models, branch_factor)| {
            let mut loc_data = data.soft_copy();
            let res = train::train(&mut loc_data, models, *branch_factor); // Train each configuration.
            // 各構成をトレーニングします。
            // 训练每个配置。
            pbar.inc(1);
            RMIStatistics::from_trained(&res) // Convert the trained RMI to statistics.
            // トレーニングされたRMIを統計に変換します。
            // 将训练的RMI转换为统计数据。
        }).collect()
}

pub fn find_pareto_efficient_configs<T: TrainingKey>(
    data: &RMITrainingData<T>, restrict: usize)
    -> Vec<RMIStatistics> {
    // Find the Pareto-efficient RMI configurations.
    // パレート効率の良いRMI構成を見つけます。
    // 找到帕累托有效的RMI配置。
    let initial_configs  = first_phase_configs(); // First phase of configurations.
    // 第一フェーズの構成。
    // 第一阶段的配置。
    let first_phase_results = measure_rmis(data, &initial_configs); // Measure the first phase.
    // 第一フェーズを測定します。
    // 测量第一阶段。

    let next_configs = second_phase_configs(&first_phase_results); // Generate second phase configurations.
    // 第二フェーズの構成を生成します。
    // 生成第二阶段的配置。
    let second_phase_results = measure_rmis(data, &next_configs); // Measure the second phase.
    // 第二フェーズを測定します。
    // 测量第二阶段。
    
    let mut final_front = pareto_front(&second_phase_results); // Find the Pareto front.
    // パレート最適曲面を見つけます。
    // 找到帕累托前沿。
    final_front = narrow_front(&final_front, restrict); // Narrow the front to the desired size.
    // パレート最適曲面を希望のサイズに狭めます。
    // 将帕累托前沿缩小到期望的大小。
    final_front.sort_by(
        |a, b| a.average_log2_error.partial_cmp(&b.average_log2_error).unwrap()
    );

    return final_front;
}
