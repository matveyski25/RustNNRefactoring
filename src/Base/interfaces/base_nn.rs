use crate::Utils::definitions_matrix::{Matrix, MatrixRef};
use ndarray::LinalgScalar;

pub type MayRes<T> = Result<T, ()>;
pub type MayErr = MayRes<()>;

pub trait Feature {
    type Context: Copy;
    fn set_context(&mut self, context: &Self::Context) -> MayRes<&mut Self>;
    fn get_context(&self) -> &Self::Context;
}

pub trait SaveManager: Feature {
    fn save(&mut self) -> MayRes<&mut Self>;
    fn load(&mut self) -> MayRes<&mut Self>;
}

pub trait Savable {
    fn set_save_manager(&mut self, manager: Box<impl SaveManager>) -> MayRes<&mut Self>;
    fn get_save_manager(&mut self) -> MayRes<&impl SaveManager>;
}

pub trait ComputeBlock<T: LinalgScalar> {
    fn set_input(&mut self, input: Matrix<T>) -> &mut Self;
    fn get_output(&self) -> MayRes<Matrix<T>>;
    fn compute(&mut self) -> MayRes<&mut Self>;
}

pub trait Computable<T: LinalgScalar> {
    fn set_compute_block(&mut self, compute_block: Box<impl ComputeBlock<T>>) -> MayRes<&mut Self>;
    fn get_compute_block(&self) -> MayRes<&impl ComputeBlock<T>>;
}

pub trait Randomizer: Feature {
    fn randomize(&mut self) -> MayRes<&mut Self>;
}

pub trait Randomizable {
    fn set_randomizer(&mut self, randomizer: Box<impl Randomizer>) -> MayRes<&mut Self>;
    fn get_randomizer(&self) -> MayRes<&impl Randomizer>;
}

pub trait Optimizer: Feature {
    fn optimize(&mut self) -> MayRes<&mut Self>;
}

pub trait Optimizable {
    fn set_optimizer(&mut self, optimizer: Box<impl Optimizer>) -> MayRes<&mut Self>;
    fn get_optimizer(&self) -> MayRes<&impl Optimizer>;
}

pub trait TranslatorMatrix<T: LinalgScalar> {
    type Input;
    type Output;
    fn convert_to_matrix(&self, input: Self::Input) -> MayRes<Matrix<T>>;
    fn convert_from_matrix(&self, matrix_ref: &MatrixRef<T>) -> MayRes<Self::Output>;
    fn convert_from_matrixv(&self, matrix: Matrix<T>) -> MayRes<Self::Output>;
}

pub trait Translatable<T: LinalgScalar> {
    type Input;
    type Output;
    fn set_translator(&mut self, translator: Box<impl TranslatorMatrix<T>>) -> MayRes<&mut Self>;
    fn get_translator(&self) -> MayRes<&impl TranslatorMatrix<T>>;
}

pub trait BaseNN<T: LinalgScalar>: Savable + Computable<T> + Translatable<T> {
    type Translator: TranslatorMatrix<T>;
    type Saver: SaveManager;
    type ComputeBlock: ComputeBlock<T>;
    fn set_input(
        &mut self,
        input: <<Self as BaseNN<T>>::Translator as TranslatorMatrix<T>>::Input,
    ) -> MayRes<&mut Self>;
    fn get_output(&self) -> <<Self as BaseNN<T>>::Translator as TranslatorMatrix<T>>::Output;
    fn inference(&self) -> MayRes<&Self>; //may will be &mut
}

pub trait TrainableComputeBlock<T: LinalgScalar>: ComputeBlock<T> + Randomizable + Optimizable {
    fn backward(&self) -> MayRes<&Self>;
    fn get_deltas_weights(&self) -> MayRes<&Self>;
}


//Trainable version trait for base trait BaseNN not be