use ndarray::{Array1, Array4, Array, arr1, Shape, Dim, Array2, IxDyn};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};
use ndarray::parallel;

#[derive(Clone, Debug)]
pub struct MatMul{
}

impl MatMul{
    pub fn new() -> MatMul{
        return MatMul{}
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> MatMul{
        return MatMul{}
    }
}

impl MatMul {
    pub fn compute(&mut self, inputs: Input) -> Output {
        return match inputs {
            Input::Tensor4List(input) => {
                let output = input.into_iter()
                    .map(|vec| {
                        let vec_tmp: Array2<f32> = vec.into_dimensionality().unwrap();
                        return vec_tmp;
                    })
                    .reduce(move |v1, v2| (v1.dot(&v2))).unwrap();
                let out_len = Vec::from(output.shape());
                return Output::TensorD(output.into_shape(IxDyn(&out_len)).unwrap());
            },
            _ => panic!("Wrong input")
        }
    }
}
