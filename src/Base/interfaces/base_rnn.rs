use num_traits::Num;

use crate::Base::interfaces::base_nn::{ComputeBlock, MayRes, TrainableComputeBlock};

pub trait ComputeBlockRNN<T: Num>: ComputeBlock<T> {
    fn all_steps_calculation(&mut self) -> MayRes<&mut Self>; //virtual
    fn step_calculation(&mut self, step: u64) -> MayRes<&mut Self>; //static
}

pub trait TrainableComputeBlockRNN<T: Num>: ComputeBlockRNN<T> + TrainableComputeBlock<T> {}
