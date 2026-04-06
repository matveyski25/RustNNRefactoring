use ndarray::LinalgScalar;

use crate::Base::interfaces::base_nn::{BaseNN, ComputeBlock, MayErr, MayRes, TrainableComputeBlock};

pub trait ComputeBlockRNN<T: LinalgScalar>: ComputeBlock<T> {
    fn all_steps_calculation(&mut self) -> MayRes<&mut Self>; //virtual
    fn step_calculation(&mut self, step: usize) -> MayErr; //static
}

pub trait TrainableComputeBlockRNN<T: LinalgScalar>: ComputeBlockRNN<T> + TrainableComputeBlock<T> {}

pub trait BaseRNN<T: LinalgScalar>: BaseNN<T>{}

//Look comment in Base::interfaces::base_nn for trainable version trait BaseRNN