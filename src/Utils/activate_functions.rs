#![allow(dead_code)]
use super::definitions_matrix::{Vector, VectorRef};
use ndarray::{Array, ArrayRef, Dimension, LinalgScalar, ScalarOperand, Zip};
use num_traits::Float;


// Обобщённая сигмоида для любой размерности
pub fn sigmoid<T: Float, D: Dimension>(x: &ArrayRef<T, D>) -> Array<T, D> {
    x.mapv(|v| T::one() / (T::one() + (-v).exp()))
}

// Обобщённый tanh
pub fn tanh<T: Float, D: Dimension>(x: &ArrayRef<T, D>) -> Array<T, D> {
    x.mapv(|v| v.tanh())
}

// Шаговая функция
pub fn step_function<T: LinalgScalar + PartialOrd, D: Dimension>(
    x: &ArrayRef<T, D>,
    step: T,
) -> Array<T, D> {
    x.mapv(|v| if v >= step { T::one() } else { T::zero() })
}

// ReLU
pub fn relu<T: LinalgScalar + PartialOrd + Copy, D: Dimension>(x: &ArrayRef<T, D>) -> Array<T, D> {
    x.mapv(|v| if v >= T::zero() { v } else { T::zero() })
}

// Leaky ReLU с поэлементным alpha
pub fn leaky_relu<T: LinalgScalar + PartialOrd + Copy, D: Dimension>(
    x: &ArrayRef<T, D>,
    alpha: &ArrayRef<T, D>,
) -> Array<T, D> {
    assert_eq!(x.shape(), alpha.shape());
    Zip::from(x)
        .and(alpha)
        .map_collect(|&xv, &av| if xv >= T::zero() { xv } else { xv * av })
}

// Leaky ReLU со скалярным alpha
pub fn leaky_relu_scalar<T: LinalgScalar + PartialOrd + Copy, D: Dimension>(
    x: &ArrayRef<T, D>,
    alpha: T,
) -> Array<T, D> {
    x.mapv(|v| if v >= T::zero() { v } else { v * alpha })
}

// Swish с поэлементным b
pub fn swish<T: Float, D: Dimension>(x: &ArrayRef<T, D>, b: &ArrayRef<T, D>) -> Array<T, D> {
    assert_eq!(x.shape(), b.shape());
    Zip::from(x)
        .and(b)
        .map_collect(|&xv, &bv| xv / (T::one() + (-xv * bv).exp()))
}

// Swish со скалярным b
pub fn swish_scalar<T: Float, D: Dimension>(x: &ArrayRef<T, D>, b: T) -> Array<T, D> {
    x.mapv(|v| v / (T::one() + (-v * b).exp()))
}

// Softmax для векторов (оставляем одномерным)
pub fn softmax<T: Float + ScalarOperand>(x: &VectorRef<T>, clamp_val: T, eps: T) -> Vector<T> {
    let x_clamped = x.mapv(|v| T::max(-clamp_val, T::min(clamp_val, v)));
    let x_max = x_clamped
        .iter()
        .fold(T::neg_infinity(), |a, &b| a.max(b));
    let exp_x = x_clamped.mapv(|xx| (xx - x_max).exp());
    let sum_exp = exp_x.sum() + eps;
    exp_x / sum_exp
}