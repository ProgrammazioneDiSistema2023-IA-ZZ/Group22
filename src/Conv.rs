use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Conv{
    autopad: String,
    dilations: Array1<i32>,
    group: u32,
    kernel_shape: Shape<Dim<[usize; 2]>>,
    pads: Array1<i32>,
    strides: Array1<i32>,
    w: Array4<f32>,
}

impl Conv{
    pub fn new(ap: Option<String>,
               dil: Option<ndarray::Array1<i32>>,
               group: Option<u32>,
               kernel_shape: Option<Shape<Dim<[usize; 2]>>>,
               pads: Option<ndarray::Array1<i32>>,
               strides: Option<ndarray::Array1<i32>>,
               w: Array4<f32>) -> Conv{
        return Conv{
            autopad: ap.unwrap_or("NOT_SET".to_string()),
            dilations: dil.unwrap_or(arr1(&[1, 1])),
            group: group.unwrap_or(1),
            kernel_shape: kernel_shape.unwrap_or(Shape::from(Dim([1, 1]))),
            pads: pads.unwrap_or(arr1(&[0, 0, 0, 0])),
            strides: strides.unwrap_or(arr1(&[1, 1])),
            w
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