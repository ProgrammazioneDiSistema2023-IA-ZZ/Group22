use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Reshape{
    allowzero: i32,
}

impl Reshape{
    pub fn new(
               allowzero: Option<i32>
               ) -> Conv{
        return Reshape{
            allowzero: allowzero.unwrap_or(0)
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
