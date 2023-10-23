use ndarray::{Array1, Array4, Array, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct MaxPool{
    autopad: String,
    ceil_mode: u32,
    dilations: Array1<i32>,
    kernel_shape: Shape<Dim<[usize; 2]>>,
    pads: Array1<i32>,
    storage_order: u32,
    strides:  Array1<i32>
}

