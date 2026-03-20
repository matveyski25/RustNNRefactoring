use derive_more::Deref;
use num_traits::Num;

use crate::Base::{
    bodies::base_nn::{
        BodyBaseNN, BodyBaseTrainableNN, BodyComputeBlock, BodyTrainableComputeBlock,
    },
    interfaces::{
        base_nn::{Optimizer, Randomizer, SaveManager, TranslatorMatrix},
        base_rnn::{ComputeBlockRNN, TrainableComputeBlockRNN},
    },
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
    pub body_trainable_compute_block: BodyTrainableComputeBlock<T, Ra, Op>,
}

#[derive(Clone, Deref)]
pub struct BodyBaseRNN<T: Num, SM: SaveManager, CB: ComputeBlockRNN<T>, Tr: TranslatorMatrix<T>>(
    pub BodyBaseNN<T, SM, CB, Tr>,
); //body_base_nn:

#[derive(Clone, Deref)]
pub struct BodyBaseTrainableRNN<
    T: Num,
    SM: SaveManager,
    CB: TrainableComputeBlockRNN<T>,
    Tr: TranslatorMatrix<T>,
> {
    pub body_base_trainable: BodyBaseTrainableNN<T, SM, CB, Tr>,
} //body_base_rnn:
