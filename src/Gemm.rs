use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Gemm{
    alpha: f32,
    beta: f32,
    transA: i32,
    transB: i32
}

impl Gemm{
    pub fn new(
        alpha: Option<f32>,
        beta: Option<f32>,
        transA: Option<i32>,
        transB: Option<i32>
    ) -> Conv{
        return Gemm{
            alpha: alpha.unwrap_or(1.0),
            beta: beta.unwrap_or(1.0),
            transA: transA.unwrap_or(0),
            transB: transB.unwrap_or(0)
        }

    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Option<Conv>{ //Change from Option to pure Conv
        //TODO Implement the method to parse from a vector of attributes
        return None;
    }

}


impl Compute for Conv{

    fn compute(&mut self, inputs: Input) -> Output {
        //TODO Implementation of the convolution
        return Output::Tensor32(Array4::from_elem((64,3,256,256), 1.5));
    }
}
