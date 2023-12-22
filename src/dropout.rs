use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Dropout{
    ratio: f32,
}

impl Dropout{
    pub fn new(
        ratio: Option<f32>
    ) -> Dropout{
        return Dropout{
            ratio: seed.unwrap_or(0.0)
        }

    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Dropout{ //Change from Option to pure Conv
        return Dropout{ratio: 0.4000000059604645};
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
