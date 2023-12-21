use std::fmt::Debug;
use ndarray::{Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayD};
use crate::add::Add;
use crate::onnx_proto3::NodeProto;


pub trait Compute {
    fn compute(&mut self, inputs: Input) -> Output;
}

impl Clone for Box<dyn Compute> {
    fn clone(&self) -> Self {
        self.clone()
    }
}

impl Debug for dyn Compute {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dyn Compute")
    }
}

/*#[derive(Clone, Debug)]
pub enum Operation{
    Convolution(Conv),
    Addition(Add)
}

impl Operation {
    pub fn get_operation(op_type: &str, node: NodeProto) -> Option<Self>{
        match op_type {
            "Conv" => Some(Convolution(Conv::parse_from_proto_node(&node.get_attribute()).unwrap())),
            "Add" => Some(Addition(Add::parse_from_proto_node(&node.get_attribute()).unwrap())),
            _ => None
        }
    }
}*/
#[derive(Clone, Debug)]
pub enum Input {
    Tensor32(Array4<f32>),
    Tensor1(Array1<f32>),
    Tensor2(Array2<f32>),
    Tensor3(Array3<f32>),
    Tensor4(Array4<f32>),
    TensorD(ArrayD<f32>), // Qui si accetta un ArrayD di f64
    Tensor32Vec(Vec<Array4<f32>>),
    Tensor4List(Vec<ArrayD<f32>>),
    Empty
}

#[derive(Clone, Debug)]
pub enum Output {
    Tensor32(Array4<f32>),
    Tensor1(Array1<f32>),
    Tensor2(Array2<f32>),
    Tensor3(Array3<f32>),
    Tensor4(Array4<f32>),
    TensorD(ArrayD<f32>)
}

