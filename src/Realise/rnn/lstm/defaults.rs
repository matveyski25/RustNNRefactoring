use derive_more::Deref;
use num_traits::Num;

use crate::{
    Base::{
        bodies::base_rnn::{BodyBaseRNN, BodyComputeBlockRNN},
        interfaces::{
            base_nn::{ComputeBlock, MayErr, MayRes, SaveManager, TranslatorMatrix},
            base_rnn::ComputeBlockRNN,
        },
    },
    Utils::definitions_matrix::{Matrix, Vector},
};

#[derive(Clone, Deref)]
struct DefaultBodyLSTM<T: Num, SM: SaveManager, CB: ComputeBlockRNN<T>, Tr: TranslatorMatrix<T>> {
    body_base_rnn: BodyBaseRNN<T, SM, CB, Tr>,
}

struct DefaultBodyComputeBlockOneH<T: Num> {
    body_compute_block_rnn: BodyComputeBlockRNN<T>,
    U: Matrix<T>, //[H x 4H]
    W: Matrix<T>, //[I x 4H]
    B: Vector<T>, //[1 x 4H]

    //LinearAlgebra::BaseMatrix<T> W_Out;
    //LinearAlgebra::BaseRowVector<T> B_Out;

    //input_state_n - [1 x I]
    n_cell_state: Vector<T>,
    n_hidden_state: Vector<T>, // [1 x H]
    tmp_f: Vector<T>,
    tmp_i: Vector<T>,
    tmp_c_bar: Vector<T>,
    tmp_o: Vector<T>,
    tmp_Z: Vector<T>,
}
impl<T: Num + Clone> DefaultBodyComputeBlockOneH<T> {
    fn setVoidNState(&mut self, hidden_size: u64) -> MayErr {
        self.n_cell_state = Vector::zeros(hidden_size as usize);
        self.n_hidden_state = Vector::zeros(hidden_size as usize);

        self.tmp_f = Vector::zeros(hidden_size as usize);
        self.tmp_i = Vector::zeros(hidden_size as usize);
        self.tmp_c_bar = Vector::<T>::zeros(hidden_size as usize);
        self.tmp_o = Vector::<T>::zeros(hidden_size as usize);

        self.tmp_Z = Vector::<T>::zeros((4 * hidden_size) as usize);

        return Ok(());
    }
}

impl<T: Num> ComputeBlockRNN<T> for DefaultBodyComputeBlockOneH<T> {
    fn all_steps_calculation(&mut self) -> MayRes<&mut Self> {
        todo!()
    }

    fn step_calculation(&mut self, step: u64) -> MayRes<&mut Self> {
        todo!()
    }
}

impl<T: Num> ComputeBlock<T> for DefaultBodyComputeBlockOneH<T> {
    fn set_input(&mut self, input: Matrix<T>) -> MayRes<&mut Self> {
        todo!()
    }

    fn get_output(&self) -> MayRes<Matrix<T>> {
        todo!()
    }

    fn compute(&mut self) -> MayRes<&mut Self> {
        todo!()
    }
}
