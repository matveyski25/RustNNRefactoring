#![allow(dead_code)]
use super::definitions_matrix::{Matrix, MatrixRef, Vector, VectorRef};
use ndarray::{ScalarOperand, Zip};
use ndarray_stats::QuantileExt;
use num_traits::{Float, Num};

pub fn step_function<T: Num + PartialOrd>(matx: &MatrixRef<T>, step: T) -> Matrix<T> {
    return matx.map(|x| if *x >= step { T::one() } else { T::zero() });
}

pub fn sigmoid<T: Float>(matx: &MatrixRef<T>) -> Matrix<T> {
    return matx.map(|x: &T| T::one() / (T::one() + (-*x).exp()));
}

pub fn tanh<T: Float>(matx: &MatrixRef<T>) -> Matrix<T> {
    return matx.map(|x: &T| x.tanh());
}

pub fn ReLU<T: Num + PartialOrd + Copy>(matx: &MatrixRef<T>) -> Matrix<T> {
    return matx.map(|x: &T| if *x >= T::zero() { *x } else { T::zero() });
}

pub fn leaky_ReLU_matrix<T: Num + PartialOrd + Copy>(
    matx: &MatrixRef<T>,
    alpha: &MatrixRef<T>,
) -> Matrix<T> {
    //if (matx.nrows() != alpha.nrows() || matx.ncols() != alphaa.ncols()) {
    //	return Err("Matrix dimensions must match");
    //}
    debug_assert!(matx.nrows() != alpha.nrows() || matx.ncols() != alpha.ncols());

    return Zip::from(matx)
        .and(alpha)
        .map_collect(|x, a| if *x >= T::zero() { *x } else { (*x) * (*a) });
}

pub fn leaky_ReLU<T: Num + PartialOrd + Copy>(matx: &MatrixRef<T>, alpha: T) -> Matrix<T> {
    return matx.map(|x| if *x >= T::zero() { *x } else { (*x) * alpha });
}

pub fn swish_matrix<T: Float>(matx: &MatrixRef<T>, b: &MatrixRef<T>) -> Matrix<T> {
    //if (matx.nrows() != b.nrows() || matx.ncols() != b.ncols()) {
    //	return Err("Matrix dimensions must match");
    //}
    debug_assert!(matx.nrows() != b.nrows() || matx.ncols() != b.ncols());

    return Zip::from(matx)
        .and(b)
        .map_collect(|x, bb| *x / ((T::one() - ((*x) * (*bb))).exp()));
}

pub fn swish<T: Float>(matx: &MatrixRef<T>, b: T) -> Matrix<T> {
    return matx.map(|x| *x / (T::one() - (*x * b)).exp());
}

pub fn softmax<T: Float + ScalarOperand>(x: &VectorRef<T>, clamp_val: T, eps: T) -> Vector<T> {
    // 1) Клэмпим входы
    let x_clamped: Vector<T> = x.map(|v| T::max(-clamp_val, T::min(clamp_val, *v)));

    // 2) Вычисляем максимум
    let x_max = x_clamped
        .iter()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    // 3) Вычисляем экспоненты от (x - max)
    let exp_x: Vector<T> = x_clamped.mapv(|xx| (xx - *x_max).exp());

    // 4) Сумма с eps
    let sum_exp: T = *exp_x.max().unwrap() + eps;

    // 5) Нормировка
    return exp_x / sum_exp;
}
