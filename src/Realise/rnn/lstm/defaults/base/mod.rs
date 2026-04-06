use crate::Base::{
    bodies::base_rnn::BodyBaseRNN,
    interfaces::{
        base_nn::{SaveManager, TranslatorMatrix},
        base_rnn::ComputeBlockRNN,
    },
};
use derive_more::{Deref, DerefMut};
use ndarray::LinalgScalar;

mod train;

#[derive(Clone, Deref, DerefMut)]
pub(crate) struct DefaultBodyLSTM<T: LinalgScalar, SM: SaveManager, CB: ComputeBlockRNN<T>, Tr: TranslatorMatrix<T>> {
    pub body_base_rnn: BodyBaseRNN<T, SM, CB, Tr>,
}

#[derive(Clone)]
pub struct DefaultLSTM<
    T: LinalgScalar, 
    SM: SaveManager, 
    CB: ComputeBlockRNN<T>, 
    Tr: TranslatorMatrix<T>
> (DefaultBodyLSTM<T, SM, CB, Tr>);


  