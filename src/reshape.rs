use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Reshape{
    allowzero: i32
}

impl Reshape{
    pub fn new(az: Option<i32>) -> Reshape{
        return Reshape{
            allowzero: az.unwrap_or(0)
        }
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Option<Reshape>{ //Change from Option to pure Conv
        //TODO Implement the method to parse from a vector of attributes
        return None;
    }

}


impl Compute for Reshape{

    fn compute(&mut self, inputs: Input) -> Output {
        //TODO Implementation of the convolution
        return Output::Tensor32(Array4::from_elem((64,3,256,256), 1.5));
    }
}

pub struct Start {
    data: Input
}

impl Start {
    pub fn new(input: Array4<f32>) -> Self {
        Start{data: Input::Tensor32(input)}
    }
}

impl Compute for Start{

    fn compute(&mut self, inputs: Input) -> Output {
        return match self.data.clone() {
            Input::Tensor32(vec) => Output::Tensor32(vec),
            _ => panic!("Wrong starting input type")
        }
    }
}