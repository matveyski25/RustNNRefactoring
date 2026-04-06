use std::marker::PhantomData;

use ndarray::LinalgScalar;

use crate::{
    Base::interfaces::base_nn::{
        ComputeBlock, Optimizer, Randomizer, SaveManager, TranslatorMatrix,
    },
    Utils::definitions_matrix::Matrix,
};

#[derive(Clone)]
pub struct BodyComputeBlock<T: LinalgScalar> {
    pub input_state: Matrix<T>, // [I x N] N - input_lenght
    pub output_state: Matrix<T>, // [1 x H]
    pub input_size: u64,
    pub output_size: u64,
}

#[derive(Clone)]
pub struct BodyBaseNN<T: LinalgScalar, SM: SaveManager, CB: ComputeBlock<T>, Tr: TranslatorMatrix<T>> {
    pub save_load_manager: Box<SM>,
    pub compute_block: Box<CB>,
    pub translator: Box<Tr>,
    _phantom_data: PhantomData<T>,
}

#[derive(Clone)]
pub struct BodyTrainableComputeBlock<T: LinalgScalar, Ra: Randomizer, Op: Optimizer> {
    randomizer: Box<Ra>,
    optimizer: Box<Op>,
    _phantom_data: PhantomData<T>,
}

#[derive(Clone)]
pub struct BodyBaseTrainableNN<
    T: LinalgScalar,
    SM: SaveManager,
    CB: ComputeBlock<T>,
    Tr: TranslatorMatrix<T>,
> {
    _phantom_data: PhantomData<T>,
    _phantom_data1: PhantomData<SM>,
    _phantom_data2: PhantomData<CB>,
    _phantom_data3: PhantomData<Tr>,
    //body_base_nn: BodyBaseNN<T, SM, CB, Tr>,
}
