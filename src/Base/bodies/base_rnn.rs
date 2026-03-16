use derive_more::Deref;
use num_traits::Num;

use crate::Base::{
    bodies::base_nn::{BodyBaseNN, BodyComputeBlock, BodyTrainableComputeBlock},
    interfaces::base_nn::{ComputeBlock, Optimizer, Randomizer, SaveManager, TranslatorMatrix},
};

#[derive(Clone, Deref)]
pub struct BodyComputeBlockRNN<T: Num> {
    #[deref]
    pub body_compute_block: BodyComputeBlock<T>,
    pub hidden_size: u64,
    pub max_steps: u64,
}

#[derive(Clone, Deref)]
pub struct BodyTrainableComputeBlockRNN<T: Num, Ra: Randomizer, Op: Optimizer> {
    pub body_compute_block_rnn: BodyComputeBlockRNN<T>,
    pub body_trainable_compute_block: BodyTrainableComputeBlock<T, Ra, Op>,
}

#[derive(Clone, Deref)]
pub struct BodyBaseRNN<T: Num, SM: SaveManager, CB: ComputeBlock<T>, Tr: TranslatorMatrix<T>>(
    BodyBaseNN<T, SM, CB, Tr>,
); //body_base_nn:

#[derive(Clone, Deref)]
pub struct BodyBaseTrainableRNN<
    T: Num,
    SM: SaveManager,
    CB: ComputeBlock<T>,
    Tr: TranslatorMatrix<T>,
>(BodyBaseRNN<T, SM, CB, Tr>); //body_base_rnn:
