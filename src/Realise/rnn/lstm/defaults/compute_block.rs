use crate::{
    Base::{
        bodies::base_rnn::BodyComputeBlockRNN,
        interfaces::{
            base_nn::{ComputeBlock, MayErr, MayRes},
            base_rnn::ComputeBlockRNN,
        },
    },
    Utils::definitions_matrix::{Matrix, Vector},
};
use derive_more::Deref;
use num_traits::Num;

#[derive(Clone, Deref)]
struct DefaultBodyComputeBlockOneH<T: Num> {
    #[deref]
    pub body_compute_block_rnn: BodyComputeBlockRNN<T>,
    pub U: Matrix<T>, //[H x 4H]
    pub W: Matrix<T>, //[I x 4H]
    pub B: Vector<T>, //[1 x 4H]

    //LinearAlgebra::BaseMatrix<T> W_Out;
    //LinearAlgebra::BaseRowVector<T> B_Out;

    //input_state_n - [1 x I]
    pub n_cell_state: Vector<T>,
    pub n_hidden_state: Vector<T>, // [1 x H]
    pub tmp_f: Vector<T>,
    pub tmp_i: Vector<T>,
    pub tmp_c_bar: Vector<T>,
    pub tmp_o: Vector<T>,
    pub tmp_Z: Vector<T>,
}
#[derive(Clone, Deref)]
struct DefaultBodyComputeBlockAllH<T: Num> {
    #[deref]
    pub body_compute_block_one_h: DefaultBodyComputeBlockOneH<T>,
    pub hidden_states_: Matrix<T>,
}

//#[derive(Clone, Deref)]
//struct DefaultBodyTrainableComputeBlockOneH {}

#[derive(Clone)]
pub struct DefaultComputeBlockOneH<T: Num>(DefaultBodyComputeBlockOneH<T>);
#[derive(Clone)]
pub struct DefaultComputeBlockAllH<T: Num>(DefaultBodyComputeBlockAllH<T>);

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
