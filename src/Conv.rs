use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::Compute;
use crate::onnx_proto3::{AttributeProto, NodeProto};

pub struct Conv{
    autopad: String,
    dilations: Array1<i32>,
    group: u32,
    kernel_shape: Shape<Dim<[usize; 2]>>,
    pads: Array1<i32>,
    strides: Array1<i32>,
    w: Array4<f32>
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
            w,
        }

    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Option<Conv>{ //Change from Option to pure Conv
        //TODO Implement the method to parse from a vector of attributes
        return None;
    }

}

impl Compute for Conv{
    type Item = Array4<f32>;

    fn compute(&mut self, inputs: Vec<&Self::Item>) -> Self::Item {
        //TODO Implementation of the convolution
        return Array4::from_elem((64,3,256,256), 1.5);
    }
}