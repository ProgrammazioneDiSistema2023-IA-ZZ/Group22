use ndarray::{Array1, Array4, Array, arr1, Shape, Dim, par_azip, s, Axis};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct MaxPool{
    autopad: String,
    ceil_mode: u32,
    dilations: Array1<i32>,
    kernel_shape: Shape<Dim<[usize; 2]>>,
    pads: Array1<i32>,
    storage_order: u32,
    strides:  Array1<i32>
}

impl MaxPool{
    pub fn new(ap: Option<String>,
               cm: Option<u32>,
               dil: Option<ndarray::Array1<i32>>,
               kernel_shape: Shape<Dim<[usize; 2]>>,
               pads: Option<ndarray::Array1<i32>>,
               st_or: Option<u32>,
               strides: Option<ndarray::Array1<i32>>,
               ) -> MaxPool{
        return MaxPool{
            autopad: ap.unwrap_or("NOT_SET".to_string()),
            ceil_mode: cm.unwrap_or(0),
            dilations: dil.unwrap_or(arr1(&[1, 1])),
            kernel_shape: kernel_shape,
            pads: pads.unwrap_or(arr1(&[0, 0, 0, 0])),
            storage_order: st_or.unwrap_or(0),
            strides: strides.unwrap_or(arr1(&[1, 1]))
        }
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Option<MaxPool>{
        //TODO Implement the method to parse from a vector of attributes
        return None;
    }
}

/*
impl Compute for MaxPool {
    fn compute(&mut self, inputs: Input) -> Output {
        match inputs {
            Input::Tensor32(input_array) => {
                let (batch, input_channels, input_height, input_width) = input_array.dim();
                let kh = self.kernel_shape.len_of(Axis(0));
                let kw = self.kernel_shape.len_of(Axis(1));

                let (sh, sw) = (self.strides[0], self.strides[1]);
                let (ph, pw) = (self.pads[0], self.pads[1]);
                let (dilation_h, dilation_w) = (self.dilations[0], self.dilations[1]);
                let ceil_mode = self.ceil_mode != 0;

                let output_height = if self.autopad == "VALID" {
                    let numerator = input_height - ((kh - 1) * dilation_h + 1) + ph;
                    if ceil_mode {
                        (numerator + sh - 1) / sh
                    } else {
                        numerator / sh
                    }
                } else {
                    if ceil_mode {
                        ((input_height - 1) / sh) + 1
                    } else {
                        (input_height - 1) / sh + 1
                    }
                };

                let output_width = if self.autopad == "VALID" {
                    let numerator = input_width - ((kw - 1) * dilation_w + 1) + pw;
                    if ceil_mode {
                        (numerator + sw - 1) / sw
                    } else {
                        numerator / sw
                    }
                } else {
                    if ceil_mode {
                        ((input_width - 1) / sw) + 1
                    } else {
                        (input_width - 1) / sw + 1
                    }
                };

                let mut output = Array::zeros((batch, input_channels, output_height, output_width));

                for b in 0..batch {
                    for c in 0..input_channels {
                        for oh in 0..output_height {
                            for ow in 0..output_width {
                                let hstart = oh * sh;
                                let hend = (oh * sh + dilation_h * (kh - 1) + 1).min(input_height);
                                let wstart = ow * sw;
                                let wend = (ow * sw + dilation_w * (kw - 1) + 1).min(input_width);
                                let input_window = input_array.slice(s![b, c, hstart..hend; dilation_h, wstart..wend; dilation_w]);
                                let max_val = input_window.iter().fold(f32::NEG_INFINITY, |acc, &x| x.max(acc));
                                output[[b, c, oh, ow]] = max_val;
                            }
                        }
                    }
                }

                println!("{}", self.kernel_shape.size());;
                //Output::Tensor32(output)
                Output::Tensor32(Array4::from_elem((64,3,256,256), 1.5))
            }
            _ => panic!("Wrong input"),
        }
        ;
    }
}

 */