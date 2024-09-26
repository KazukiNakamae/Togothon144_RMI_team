// < begin copyright >
// Copyright Ryan Marcus 2020
//
// See root directory of this project for license terms.
//
// < end copyright >

// Import the memmap crate for memory-mapped file operations.
// メモリマップされたファイル操作のためにmemmapクレートをインポートします。
// 导入 memmap crate，用于内存映射文件操作。
use memmap::MmapOptions;
// Import necessary modules from the RMI library.
// RMIライブラリから必要なモジュールをインポートします。
// 从 RMI 库导入必要的模块。
use rmi_lib::{RMITrainingData, RMITrainingDataIteratorProvider, KeyType, U512};
// Import byte order handling for reading data in little-endian format.
// リトルエンディアン形式でデータを読み取るためのバイト順序処理をインポートします。
// 导入字节顺序处理，用于以小端格式读取数据。
use byteorder::{LittleEndian, ReadBytesExt};
// Standard library imports for file operations and conversions.
// ファイル操作や変換のための標準ライブラリをインポートします。
// 导入标准库，用于文件操作和转换。
use std::fs::File;
use std::convert::TryInto;

// Import the Integer type from the `rug` crate for handling large integers.
// `rug`クレートから大きな整数を扱うためにInteger型をインポートします。
// 从 `rug` crate 导入 Integer 类型，用于处理大整数。

// Define an enum for supported data types (e.g., UINT64, UINT512, FLOAT64).
// サポートされているデータ型を定義する列挙型です（例: UINT64, UINT512, FLOAT64）。
// 定义一个枚举用于支持的数据类型（例如 UINT64、UINT512、FLOAT64）。
pub enum DataType {
    UINT64,
    UINT128,
    UINT32,
    UINT512,
    FLOAT64
}

// Define a struct to map a memory-mapped file for u64 data.
// u64データのためのメモリマップファイルをマッピングするための構造体を定義します。
// 定义一个结构体，用于映射 u64 数据的内存映射文件。
struct SliceAdapterU64 {
    data: memmap::Mmap,
    length: usize
}

// Implement the `RMITrainingDataIteratorProvider` trait for u64 data.
// u64データのために`RMITrainingDataIteratorProvider`トレイトを実装します。
// 为 u64 数据实现 `RMITrainingDataIteratorProvider` 特性。
impl RMITrainingDataIteratorProvider for SliceAdapterU64 {
    type InpType = u64;

    // Provide an iterator over the cumulative distribution function (CDF) for the data.
    // データに対して累積分布関数（CDF）のイテレータを提供します。
    // 提供一个迭代器，用于数据的累积分布函数 (CDF)。
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (Self::InpType, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }

    // Retrieve the data item at a specific index.
    // 特定のインデックスでデータ項目を取得します。
    // 获取特定索引处的数据项。
    fn get(&self, idx: usize) -> Option<(Self::InpType, usize)> {
        if idx >= self.length { return None; };
        let mi = (u64::from_le_bytes((&self.data[8 + idx * 8..8 + (idx + 1) * 8])
                                    .try_into().unwrap()) >> 0) << 0;
        return Some((mi.into(), idx));
    }

    // Return the type of key used (in this case, u64).
    // 使用されるキーの型を返します（この場合はu64）。
    // 返回使用的键类型（在此情况下为 u64）。
    fn key_type(&self) -> KeyType {
        KeyType::U64
    }

    // Return the total number of data items.
    // データ項目の総数を返します。
    // 返回数据项的总数。
    fn len(&self) -> usize { self.length }
}

// Define a similar struct for u32 data and implement the same trait.
// u32データのための類似の構造体を定義し、同じトレイトを実装します。
// 为 u32 数据定义一个类似的结构体并实现相同的特性。
struct SliceAdapterU32 {
    data: memmap::Mmap,
    length: usize
}

impl RMITrainingDataIteratorProvider for SliceAdapterU32 {
    type InpType = u32;
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (Self::InpType, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }

    fn get(&self, idx: usize) -> Option<(Self::InpType, usize)> {
        if idx >= self.length { return None; };
        let mi = (&self.data[8 + idx * 4..8 + (idx + 1) * 4])
            .read_u32::<LittleEndian>().unwrap().into();
        return Some((mi, idx));
    }

    fn key_type(&self) -> KeyType {
        KeyType::U32
    }

    fn len(&self) -> usize { self.length }
}

// Define a struct and implement the same trait for handling U512 data.
// U512データを扱うための構造体を定義し、同じトレイトを実装します。
// 为处理 U512 数据定义一个结构体并实现相同的特性。
struct SliceAdapterU512 {
    data: memmap::Mmap,
    length: usize
}

impl RMITrainingDataIteratorProvider for SliceAdapterU512 {
    type InpType = U512;

    // Provide an iterator for the CDF of U512 data.
    // U512データのCDFのためのイテレータを提供します。
    // 提供 U512 数据的 CDF 的迭代器。
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (Self::InpType, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }

    // Retrieve the U512 data at a specific index.
    // 特定のインデックスでU512データを取得します。
    // 获取特定索引处的 U512 数据。
    fn get(&self, idx: usize) -> Option<(Self::InpType, usize)> {
        if idx >= self.length { return None; };
        let mut source: [u64; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
        for i_ in 0..8 {
            source[i_] = (&self.data[8 + idx * 64 + i_ * 8..8 + idx * 64 + (i_ + 1) * 8])
                .read_u64::<LittleEndian>().unwrap().into();
        }
        return Some((U512(source), idx));
    }

    fn key_type(&self) -> KeyType {
        KeyType::U512
    }

    fn len(&self) -> usize { self.length }
}

// Define a struct and implement the same trait for f64 data.
// f64データを扱うための構造体を定義し、同じトレイトを実装します。
// 为处理 f64 数据定义一个结构体并实现相同的特性。
struct SliceAdapterF64 {
    data: memmap::Mmap,
    length: usize
}

impl RMITrainingDataIteratorProvider for SliceAdapterF64 {
    type InpType = f64;
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (Self::InpType, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }

    fn get(&self, idx: usize) -> Option<(Self::InpType, usize)> {
        if idx >= self.length { return None; };
        let mi = (&self.data[8 + idx * 8..8 + (idx + 1) * 8])
            .read_f64::<LittleEndian>().unwrap().into();
        return Some((mi, idx));
    }

    fn key_type(&self) -> KeyType {
        KeyType::F64
    }

    fn len(&self) -> usize { self.length }
}

// Define a struct and implement the same trait for u128 data.
// u128データを扱うための構造体を定義し、同じトレイトを実装します。
// 为处理 u128 数据定义一个结构体并实现相同的特性。
struct SliceAdapterU128 {
    data: memmap::Mmap,
    length: usize
}

impl RMITrainingDataIteratorProvider for SliceAdapterU128 {
    type InpType = u128;
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (Self::InpType, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }

    fn get(&self, idx: usize) -> Option<(Self::InpType, usize)> {
        if idx >= self.length { return None; };
        let mi = (u128::from_le_bytes((&self.data[8 + idx * 16..8 + (idx + 1) * 16])
                                    .try_into().unwrap()) >> 0) << 0;
        return Some((mi.into(), idx));
    }

    fn key_type(&self) -> KeyType {
        KeyType::U128
    }

    fn len(&self) -> usize { self.length }
}

// Define an enum to hold different types of memory-mapped RMI training data.
// 異なるタイプのメモリマップされたRMIトレーニングデータを保持するための列挙型を定義します。
// 定义一个枚举，用于存储不同类型的内存映射 RMI 训练数据。
pub enum RMIMMap {
    UINT64(RMITrainingData<u64>),
    UINT32(RMITrainingData<u32>),
    UINT512(RMITrainingData<U512>),
    UINT128(RMITrainingData<u128>),
    FLOAT64(RMITrainingData<f64>)
}

// Macro for dynamically dispatching the training function based on the data type.
// データ型に基づいてトレーニング関数を動的にディスパッチするためのマクロ。
// 宏，用于根据数据类型动态分派训练函数。
macro_rules! dynamic {
    ($funcname: expr, $data: expr $(, $p: expr )*) => {
        match $data {
            load::RMIMMap::UINT64(mut x) => $funcname(&mut x, $($p),*),
            load::RMIMMap::UINT32(mut x) => $funcname(&mut x, $($p),*),
            load::RMIMMap::UINT128(mut x) => $funcname(&mut x, $($p),*),
            load::RMIMMap::UINT512(mut x) => $funcname(&mut x, $($p),*),
            load::RMIMMap::FLOAT64(mut x) => $funcname(&mut x, $($p),*),
        }
    }
}

// Implementation for copying and converting RMIMMap types.
// RMIMMap型のコピーと変換の実装。
// 实现 RMIMMap 类型的复制和转换。
impl RMIMMap {
    pub fn soft_copy(&self) -> RMIMMap {
        match self {
            RMIMMap::UINT64(x) => RMIMMap::UINT64(x.soft_copy()),
            RMIMMap::UINT32(x) => RMIMMap::UINT32(x.soft_copy()),
            RMIMMap::UINT128(x) => RMIMMap::UINT128(x.soft_copy()),
            RMIMMap::UINT512(x) => RMIMMap::UINT512(x.soft_copy()),
            RMIMMap::FLOAT64(x) => RMIMMap::FLOAT64(x.soft_copy()),
        }
    }

    pub fn into_u64(self) -> Option<RMITrainingData<u64>> {
        match self {
            RMIMMap::UINT64(x) => Some(x),
            _ => None
        }
    }
}

// Load data from a file and create the appropriate RMIMMap based on the data type.
// ファイルからデータを読み込み、データ型に基づいて適切なRMIMMapを作成します。
// 从文件加载数据，并根据数据类型创建相应的 RMIMMap。
pub fn load_data(filepath: &str,
                 dt: DataType) -> (usize, RMIMMap) {
    // Open the file at the specified path.
    // 指定されたパスでファイルを開きます。
    // 在指定路径打开文件。
    let fd = File::open(filepath).unwrap_or_else(|_| {
        panic!("Unable to open data file at {}", filepath)
    });

    // Memory map the file for reading.
    // 読み取りのためにファイルをメモリマップします。
    // 为读取映射文件到内存。
    let mmap = unsafe { MmapOptions::new().map(&fd).unwrap() };
    let num_items = (&mmap[0..8]).read_u64::<LittleEndian>().unwrap() as usize;

    // Match the data type and create the appropriate RMIMMap variant.
    // データ型を一致させ、適切なRMIMMapのバリアントを作成します。
    // 匹配数据类型并创建相应的 RMIMMap 变体。
    let rtd = match dt {
        DataType::UINT64 =>
            RMIMMap::UINT64(RMITrainingData::new(Box::new(
                SliceAdapterU64 { data: mmap, length: num_items }
            ))),
        DataType::UINT32 =>
            RMIMMap::UINT32(RMITrainingData::new(Box::new(
                SliceAdapterU32 { data: mmap, length: num_items }
            ))),
        DataType::UINT128 =>
            RMIMMap::UINT128(RMITrainingData::new(Box::new(
                SliceAdapterU128 { data: mmap, length: num_items }
            ))),
        DataType::UINT512 =>
            RMIMMap::UINT512(RMITrainingData::new(Box::new(
                SliceAdapterU512 { data: mmap, length: num_items }
            ))),
        DataType::FLOAT64 =>
            RMIMMap::FLOAT64(RMITrainingData::new(Box::new(
                SliceAdapterF64 { data: mmap, length: num_items }
            )))
    };

    return (num_items, rtd);
}
