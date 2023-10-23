use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Add{
}

impl Add{
    pub fn new(constant: f32) -> Add{
        return Add{ }

    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Add{
        return Add{}
    }

}

impl Compute for Add{
    fn compute(&mut self, inputs: Input) -> Output {
        return match inputs {
            Input::Tensor32Vec(input) => {
                let output = input.into_iter()
                    .reduce(move |v1, v2| (v1 + v2)).unwrap();
                return Output::Tensor32(output.clone());
            },
            _ => panic!("Wrong input")
        }
    }
}

#[derive(Clone, Debug)]
pub struct AddToTryGraph {
    costant: f32
}

impl AddToTryGraph {
    pub fn new(costant: f32) -> AddToTryGraph {
        return AddToTryGraph { costant}

    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> AddToTryGraph {
        return AddToTryGraph { costant: 1.0}
    }

}

impl Compute for AddToTryGraph {
    fn compute(&mut self, inputs: Input) -> Output {
        return match inputs {
            Input::Tensor32(input) => Output::Tensor32(input + self.costant),
            Input::Tensor32Vec(input) => {
                let output = input.into_iter()
                    .reduce(move |v1, v2| (v1 + v2)).unwrap();
                return Output::Tensor32(output.clone());
            },
            _ => panic!("Wrong input")
        }
    }
}