use crate::{Base::{
    bodies::base_rnn::BodyBaseTrainableRNN,
    interfaces::{
        base_nn::{SaveManager, TranslatorMatrix},
        base_rnn::TrainableComputeBlockRNN,
    },
}, Realise::rnn::lstm::defaults::base::DefaultBodyLSTM};
use derive_more::{Deref, DerefMut};
use ndarray::LinalgScalar;

#[derive(Clone, Deref, DerefMut)]
pub(crate) struct DefaultBodyTrainableLSTM<
    T: LinalgScalar,
    SM: SaveManager,
    CB: TrainableComputeBlockRNN<T>,
    Tr: TranslatorMatrix<T>,
> {
    pub body_base_trainable_rnn: BodyBaseTrainableRNN<T, SM, CB, Tr>,
}
pub struct DefaultTrainableLSTM<
    T: LinalgScalar,
    SM: SaveManager,
    CB: TrainableComputeBlockRNN<T>,
    Tr: TranslatorMatrix<T>,
> (DefaultBodyLSTM<T, SM, CB, Tr>, DefaultBodyTrainableLSTM<T, SM, CB, Tr>,);

