use crate::{Realise::rnn::lstm::defaults::compute_block::one_h::DefaultBodyComputeBlockOneH, Utils::definitions_matrix::Matrix};
use derive_more::{Deref, DerefMut};
use ndarray::LinalgScalar;

#[derive(Clone, Deref, DerefMut)]
pub(crate) struct DefaultBodyComputeBlockAllH<T: LinalgScalar> {
    #[deref]
    #[deref_mut]
    pub body_compute_block_one_h: DefaultBodyComputeBlockOneH<T>,
    pub hidden_states: Matrix<T>,
}

#[derive(Clone)]
pub struct DefaultComputeBlockForBiLSTM<T: LinalgScalar>(DefaultBodyComputeBlockAllH<T>);
