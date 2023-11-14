use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Dropout{
    seed: i32,
}

impl Dropout{
    pub fn new(
        seed: Option<i32>
    ) -> Dropout{
        return Dropout{
            seed: seed.unwrap_or(0)//i don't know what the default value is
        }

    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Option<Dropout>{ //Change from Option to pure Conv
        //TODO Implement the method to parse from a vector of attributes
        return None;
    }

}


impl Compute for Dropout{

    fn compute(&mut self, inputs: Input) -> Output {
        let out = match inputs{
            Input::TensorD(array) => array,
            _ => panic!("Wrong input")
        };
        return Output::TensorD(out);
    }
}
