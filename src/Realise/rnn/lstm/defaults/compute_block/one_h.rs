use std::ops::AddAssign;

use derive_more::{Deref, DerefMut};

use ndarray::{LinalgScalar, s};
use num_traits::Float;
use crate::{
    Base::{
        bodies::base_rnn::BodyComputeBlockRNN,
        interfaces::{base_nn::{ComputeBlock, MayErr, MayRes}, base_rnn::ComputeBlockRNN},
    },
    Utils::{definitions_matrix::{Matrix, MatrixView, Vector, VectorView}, activate_functions::{sigmoid, tanh}},
};

#[derive(Clone, Deref, DerefMut)]
pub(crate) struct DefaultBodyComputeBlockOneH<T: LinalgScalar> {
    #[deref]
    #[deref_mut]
    pub body_compute_block_rnn: BodyComputeBlockRNN<T>,
    pub U: Matrix<T>,
    pub W: Matrix<T>,
    pub B: Vector<T>, 
    //LinearAlgebra::BaseMatrix<T> W_Out;
    //LinearAlgebra::BaseRowVector<T> B_Out;

    pub n_cell_state: Vector<T>,
    pub n_hidden_state: Vector<T>,
    pub tmp_f: Vector<T>,
    pub tmp_i: Vector<T>,
    pub tmp_c_bar: Vector<T>,
    pub tmp_o: Vector<T>,
    pub tmp_Z: Vector<T>,
    //x_t - [I x 1] col
    //h_t - [H x 1] col
    //c_t - [H x 1] col
    //U - [4H x H]
    //W - [4H x I]
    //B - [4H x 1]
    //W_out - [O x H]
    //B_out - [O x 1]
    //y_t - [O x 1]
}

#[derive(Clone)]
pub struct DefaultComputeBlock<T: LinalgScalar>(DefaultBodyComputeBlockOneH<T>);

impl<T: LinalgScalar> DefaultBodyComputeBlockOneH<T> {
    fn set_void_nstate(&mut self, hidden_size: usize) -> MayErr {
        self.n_cell_state = Vector::zeros(hidden_size);
        self.n_hidden_state = Vector::zeros(hidden_size);

        self.tmp_f = Vector::zeros(hidden_size);
        self.tmp_i = Vector::zeros(hidden_size);
        self.tmp_c_bar = Vector::<T>::zeros(hidden_size);
        self.tmp_o = Vector::<T>::zeros(hidden_size);

        self.tmp_Z = Vector::<T>::zeros((4 * hidden_size));

        return Ok(());
    }
    //fn new_in_void(hidden_size: u64) -> MayRes<Self> {
    //    return Ok(Self {
    //        n_cell_state: Vector::zeros(hidden_size as usize),
    //        n_hidden_state: Vector::zeros(hidden_size as usize),
    //        tmp_f: Vector::zeros(hidden_size as usize),
    //        tmp_i: Vector::zeros(hidden_size as usize),
    //        tmp_c_bar: Vector::<T>::zeros(hidden_size as usize),
    //        tmp_o: Vector::<T>::zeros(hidden_size as usize),
    //        tmp_Z: Vector::<T>::zeros((4 * hidden_size) as usize),
    //        
    //    });
    //}
}
  
impl<T: LinalgScalar + AddAssign + Float> ComputeBlockRNN<T> for DefaultComputeBlock<T> {
    fn all_steps_calculation(&mut  self) -> MayRes<&mut Self> {
        let steps = std::cmp::max(self.0.max_steps, self.0.input_state.nrows());
        for i in 0..steps {
            self.step_calculation(i);
        }
        todo!()
    }

    fn step_calculation(&mut self, step: usize) -> MayErr {
        let H: usize = self.0.hidden_size;

		let c_n_l: VectorView<T> = self.0.n_cell_state.view(); //col
		let h_n_l: VectorView<T> = self.0.n_hidden_state.view(); //col

		let W: MatrixView<T> = self.0.W.view();
		let U: MatrixView<T> = self.0.U.view();
		let B: VectorView<T> = self.0.B.view(); //col

		let x_n: VectorView<T> = self.0.input_state.column(step); //col

		self.0.tmp_Z = W.dot(&x_n) + U.dot(&h_n_l);
		self.0.tmp_Z += &B; //[4H x 1]

		self.0.tmp_f = sigmoid(&self.0.tmp_Z.slice(s![..H]));
		self.0.tmp_i = sigmoid(&self.0.tmp_Z.slice(s![H..2*H]));
		self.0.tmp_c_bar = tanh(&self.0.tmp_Z.slice(s![2*H..3*H]));
		self.0.tmp_o = sigmoid(&self.0.tmp_Z.slice(s![3*H..]));

		let new_c_n: Vector<T> = (&self.0.tmp_f * &c_n_l) + (&self.0.tmp_i *
			&self.0.tmp_c_bar); //col
		let new_h_n: Vector<T> = &self.0.tmp_o * tanh(&new_c_n); //col


		self.0.n_cell_state = new_c_n;
		self.0.n_hidden_state = new_h_n;
        
        return Ok(());
    }
}

impl<T: LinalgScalar + AddAssign + Float> ComputeBlock<T> for DefaultComputeBlock<T> {
    fn set_input(&mut self, input: Matrix<T>) -> &mut Self {
        self.0.input_state = input;
        return self;
    }

    fn get_output(&self) -> MayRes<Matrix<T>> {
        let output: Matrix<T> = self.0.output_state.clone();
        return Ok(output);
    }

    fn compute(&mut self) -> MayRes<&mut Self> {
        let result = self.all_steps_calculation();
        self.0.output_state = self.0.n_hidden_state.clone();

        todo!()
    }
}