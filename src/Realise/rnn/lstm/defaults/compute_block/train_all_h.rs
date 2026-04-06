use crate::{Base::interfaces::base_nn::{Optimizer, Randomizer}, Realise::rnn::lstm::defaults::compute_block::{all_h::DefaultBodyComputeBlockAllH, train_one_h::DefaultBodyTrainableComputeBlockOneH}};
use derive_more::{Deref, DerefMut};
use ndarray::LinalgScalar;

#[derive(Clone, Deref, DerefMut)]
pub(crate) struct DefaultBodyTrainableComputeBlockAllH<T: LinalgScalar, Ra: Randomizer, Op: Optimizer> (pub DefaultBodyTrainableComputeBlockOneH<T, Ra, Op>);

#[derive(Clone)]
pub struct DefaultTrainableComputeBlockForBiLSTM<T: LinalgScalar, Ra: Randomizer, Op: Optimizer> {
    base: DefaultBodyComputeBlockAllH<T>, 
    train: DefaultBodyTrainableComputeBlockAllH<T, Ra, Op>
}
