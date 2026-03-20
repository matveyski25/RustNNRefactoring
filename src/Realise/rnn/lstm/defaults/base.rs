use crate::Base::{
    bodies::base_rnn::{BodyBaseRNN, BodyBaseTrainableRNN},
    interfaces::{
        base_nn::{SaveManager, TranslatorMatrix},
        base_rnn::{ComputeBlockRNN, TrainableComputeBlockRNN},
    },
};
use derive_more::Deref;
use num_traits::Num;

#[derive(Clone, Deref)]
struct DefaultBodyLSTM<T: Num, SM: SaveManager, CB: ComputeBlockRNN<T>, Tr: TranslatorMatrix<T>> {
    body_base_rnn: BodyBaseRNN<T, SM, CB, Tr>,
}

#[derive(Clone, Deref)]
struct DefaultBodyTrainableLSTM<
    T: Num,
    SM: SaveManager,
    CB: TrainableComputeBlockRNN<T>,
    Tr: TranslatorMatrix<T>,
> {
    body_base_trainable_rnn: BodyBaseTrainableRNN<T, SM, CB, Tr>,
}
