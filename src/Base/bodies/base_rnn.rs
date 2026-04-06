use derive_more::{Deref, DerefMut};
use ndarray::LinalgScalar;

use crate::Base::{
    bodies::base_nn::{
        BodyBaseNN, BodyBaseTrainableNN, BodyComputeBlock, BodyTrainableComputeBlock,
    },
    interfaces::{
        base_nn::{Optimizer, Randomizer, SaveManager, TranslatorMatrix},
        base_rnn::{ComputeBlockRNN, TrainableComputeBlockRNN},
    },
};

#[derive(Clone, Deref, DerefMut)]
pub struct BodyComputeBlockRNN<T: LinalgScalar> {
    #[deref]
    #[deref_mut]
    pub body_compute_block: BodyComputeBlock<T>,
    pub hidden_size: usize,
    pub max_steps: usize,
}

#[derive(Clone, Deref, DerefMut)]
pub struct BodyTrainableComputeBlockRNN<T: LinalgScalar, Ra: Randomizer, Op: Optimizer> {
    pub body_trainable_compute_block: BodyTrainableComputeBlock<T, Ra, Op>,
}

#[derive(Clone, Deref, DerefMut)]
pub struct BodyBaseRNN<T: LinalgScalar, SM: SaveManager, CB: ComputeBlockRNN<T>, Tr: TranslatorMatrix<T>>(
    pub BodyBaseNN<T, SM, CB, Tr>,
); //body_base_nn:

#[derive(Clone, Deref, DerefMut)]
pub struct BodyBaseTrainableRNN<
    T: LinalgScalar,
    SM: SaveManager,
    CB: TrainableComputeBlockRNN<T>,
    Tr: TranslatorMatrix<T>,
> {
    pub body_base_trainable: BodyBaseTrainableNN<T, SM, CB, Tr>,
} //body_base_rnn:
