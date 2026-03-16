use std::marker::PhantomData;

use crate::{
    Base::interfaces::base_nn::{
        ComputeBlock, Optimizer, Randomizer, SaveManager, TranslatorMatrix,
    },
    Utils::definitions_matrix::Matrix,
};
use derive_more::Deref;
use num_traits::Num;

#[derive(Clone)]
pub struct BodyComputeBlock<T: Num> {
    pub input_state: Matrix<T>,
    pub output_state_: Matrix<T>,
    pub input_size_: u64,
    pub output_size_: u64,
}

#[derive(Clone)]
pub struct BodyBaseNN<T: Num, SM: SaveManager, CB: ComputeBlock<T>, Tr: TranslatorMatrix<T>> {
    pub save_load_manager_: Box<SM>,
    pub compute_block_: Box<CB>,
    pub translator_: Box<Tr>,
    _phantom_data: PhantomData<T>,
}

#[derive(Clone, Deref)]
pub struct BodyTrainableComputeBlock<T: Num, Ra: Randomizer, Op: Optimizer> {
    #[deref]
    body_compute_block: BodyComputeBlock<T>,
    randomizer_: Box<Ra>,
    optimizer_: Box<Op>,
}

#[derive(Clone, Deref)]
pub struct BodyBaseTrainableNN<
    T: Num,
    SM: SaveManager,
    CB: ComputeBlock<T>,
    Tr: TranslatorMatrix<T>,
> {
    body_base_nn: BodyBaseNN<T, SM, CB, Tr>,
}
