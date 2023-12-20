use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};
use std::cmp::max;
use ndarray::ArrayD;

#[derive(Clone, Debug)]
pub struct Conv{
    autopad: String,
    dilations: Array1<i32>,
    group: u32,
    kernel_shape: Shape<Dim<[usize; 2]>>,
    pads: Array1<i32>,
    strides: Array1<i32>,
}

impl Conv{
    pub fn new(ap: Option<String>,
               dil: Option<ndarray::Array1<i32>>,
               group: Option<u32>,
               kernel_shape: Option<Shape<Dim<[usize; 2]>>>,
               pads: Option<ndarray::Array1<i32>>,
               strides: Option<ndarray::Array1<i32>>, ) -> Conv{
        return Conv{
            autopad: ap.unwrap_or("NOT_SET".to_string()),
            dilations: dil.unwrap_or(arr1(&[1, 1])),
            group: group.unwrap_or(1),
            kernel_shape: kernel_shape.unwrap_or(Shape::from(Dim([1, 1]))),
            pads: pads.unwrap_or(arr1(&[0, 0, 0, 0])),
            strides: strides.unwrap_or(arr1(&[1, 1]))
        }

    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Conv{ //Change from Option to pure Conv
        //TODO Implement the method to parse from a vector of attributes
        let mut conv_tmp = Conv::new(None, None, None,
                                     None, None, None);
        for attr in attributes.iter(){
            match attr.name{
                "dilations" => {
                    conv_tmp.dilations = Array1::from(attr.ints);
                }
            }
        }
        return conv_tmp;
    }

}


impl Compute for Conv{

    fn compute(&mut self, inputs: Input) -> Output {
        let autopad = self.autopad.clone();
        let dilations = self.dilations.clone();
        let group = self.group;
        let kernel_shape = self.kernel_shape;
        let pads = self.pads.clone();
        let strides = self.strides.clone();

        let mut vec = match inputs {
            Input::Tensor4List(vec_array) => vec_array,
            _ => panic!("Input is not a vector")
        };
        let mut x = vec[0];
        let mut w1 = vec[1];
        let mut b = vec[2];

        // Retrieve input dimensions
        //let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);

        let (mut n, mut c, mut h, mut w) = (0, 0, 0, 0);
        for tensor in x.iter() {
            // Get the size of each dimension
            let shape = tensor.shape();
            match shape.len() {
                4 => {
                    n = shape[0].clone();
                    c = shape[1].clone();
                    h = shape[2].clone();
                    w = shape[3].clone();
                },
                _ => panic!("Unexpected number of dimensions in the tensor"),
            }
        }

        // Retrieve weight dimensions

        let (m, _, kh, kw) = {
            // Get the size of each dimension
            let shape = w1.shape();
            match shape.len() {
                4 => (shape[0].clone(), shape[1].clone(), shape[2].clone(), shape[3].clone()),
                _ => panic!("Unexpected number of dimensions in the array"),
            }
        };
        // Calculate output dimensions
        //let oh = self.calculate_output_size(h, kh);
        //let ow = self.calculate_output_size(w, kw);

        let oh = {
            match autopad.as_str() {
                "SAME_UPPER" | "SAME_LOWER" => {
                    // Convert usize to i32
                    let output_size = (h + strides[0] as usize - 1) / strides[0] as usize;
                    max(output_size, kh)
                }
                _ => ((h - kh + 2 * pads[0] as usize) / strides[0] as usize) + 1,
            }
        };

        let ow = {
            match autopad.as_str() {
                "SAME_UPPER" | "SAME_LOWER" => {
                    let output_size = (w + strides[0] as usize - 1) / strides[0] as usize;
                    max(output_size, kw)
                }
                _ => ((w - kw + 2 * pads[0] as usize) / strides[0] as usize) + 1,
            }
        };

        // Initialize output tensor
        let mut y = Array4::<f32>::zeros((n, m, oh, ow));

        // Convolution computation
        for ni in 0..n {
            for mi in 0..m {
                for hi in 0..oh {
                    for wi in 0..ow {
                        let mut sum = 0.0;

                        for ci in 0..c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let xi = hi * strides[0] as usize + ki - pads[0] as usize;
                                    let xj = wi * strides[1] as usize + kj - pads[2] as usize;

                                    if xi < h as usize && xj < w as usize {
                                        // Access individual ArrayD within the Vec
                                        let x_tensor: &ArrayD<f32> = &x[ni];
                                        let w1_tensor = &w1;

                                        sum += x_tensor[[ci, xi, xj]] * w1_tensor[[mi, ci, ki, kj]];
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias) = &b {
                            sum += bias[mi];
                        }

                        y[[ni as usize, mi as usize, hi as usize, wi as usize]] = sum;
                        //the output tensor will have dimensions (n, m, oh, ow) where n is the number
                        //of examples in the batch; m is the number of output channels; oh is the
                        //output height; ow is the output width
                    }
                }
            }
        }

        Output::TensorD(y.into_dyn())
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

/*
notes about the attributes:
-auto_pad: ...When set to SAME_UPPER or SAME_LOWER, the input is padded to ensure that for each
 axis i, the output shape output_shape[i] is approximately equal to ceil(input_shape[i] / strides[i])...

The operation ceil(input_shape[i] / strides[i]) involves two mathematical operations: division and ceiling rounding.

-Division (input_shape[i] / strides[i]): It calculates the result of dividing the value of
input_shape[i] by the value of strides[i]. This operation represents the number of steps or
intervals along a specific axis (indexed by i) that the convolutional or pooling operation takes.
*strides: The sliding window moves with a specified stride, determining how much it shifts over
the input after each operation -> the stride represents how many steps you go further before repeating
a certain operation (Max Pooling for instance)

-Ceiling Rounding (ceil(...)): The ceil function rounds the result of the division operation to
the smallest integer greater than or equal to the result. In the context of neural networks and
convolutional operations, this rounding is often used to ensure that the output size is sufficient
 to cover the entire input, even when the division result is not an integer.

N.B.In the context of a convolutional neural network (CNN), the convolution kernel (also referred
to as a filter) is a small matrix applied to an input to produce a feature map. It plays a crucial
role in the convolution operation, which is a fundamental operation in CNNs.

The convolution operation involves sliding a kernel window over the input data (image or feature map)
and computing the element-wise multiplication between the values in the kernel and the overlapping
input values. The results are then summed up to produce a single value in the output feature map.
This process is repeated for every position of the kernel window, resulting in the entire output feature map
 */

