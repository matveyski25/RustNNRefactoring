#![allow(dead_code)]
extern crate ndarray;
extern crate num_traits;

use ndarray::{ArcArray1, ArcArray2};
use ndarray::{Array1, Array2};
use ndarray::{ArrayRef1, ArrayRef2};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{ArrayViewMut1, ArrayViewMut2};
//Can needed in future - Scalar
//use ndarray::{Array0, ArrayRef0, ArrayView0, ArrayViewMut0};
//pub type Scalar<T> = Array0;
//pub type ScalarRef<T> = ArrayRef0;
//pub type ScalarView<T> = ArrayView0;
//pub type ScalarViewMut<T> = ArrayViewMut0;

pub type Vector<T> = Array1<T>;
pub type Matrix<T> = Array2<T>;
pub type ArcVector<T> = ArcArray1<T>;
pub type ArcMatrix<T> = ArcArray2<T>;
pub type VectorRef<T> = ArrayRef1<T>;
pub type MatrixRef<T> = ArrayRef2<T>;
pub type VectorView<'a, T> = ArrayView1<'a, T>;
pub type MatrixView<'a, T> = ArrayView2<'a, T>;
pub type VectorViewMut<'a, T> = ArrayViewMut1<'a, T>;
pub type MatrixViewMut<'a, T> = ArrayViewMut2<'a, T>;
