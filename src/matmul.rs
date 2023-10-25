use ndarray::{Array1, Array4, Array, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};
use ndarray::parallel;

#[derive(Clone, Debug)]
pub struct MatMul{
}

impl MatMul{
    pub fn new(constant: f32) -> MatMul{
        return MatMul{}
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> MatMul{
        return MatMul{}
    }
}

impl MatMul {
    pub fn compute(&mut self, inputs: Input) -> Output {
        return match inputs {
            Input::Tensor2Vec(input) => {
                let output = input.into_iter()
                    .reduce(move |v1, v2| (v1.dot(&v2))).unwrap();
                return Output::Tensor2(output.clone());
            },
            _ => panic!("Wrong input")
        }
    }
}
