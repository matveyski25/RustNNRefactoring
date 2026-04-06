use crate::{Base::{
    bodies::base_rnn::BodyTrainableComputeBlockRNN,
    interfaces::base_nn::{Optimizer, Randomizer},
}, Realise::rnn::lstm::defaults::compute_block::one_h::DefaultBodyComputeBlockOneH};
use derive_more::{Deref, DerefMut};
use ndarray::LinalgScalar;

#[derive(Clone)]
struct TrainState {

}


#[derive(Clone, Deref, DerefMut)]
pub(crate) struct DefaultBodyTrainableComputeBlockOneH<T: LinalgScalar, Ra: Randomizer, Op: Optimizer> {
    #[deref]
    #[deref_mut]
    pub body_trainable_compute_block_rnn: BodyTrainableComputeBlockRNN<T, Ra, Op>,
    pub train_states: Vec<TrainState>,
}

#[derive(Clone)]
pub struct DefaultTrainableComputeBlock<T: LinalgScalar, Ra: Randomizer, Op: Optimizer> {
    base: DefaultBodyComputeBlockOneH<T>, 
    train: DefaultBodyTrainableComputeBlockOneH<T, Ra, Op>
}
