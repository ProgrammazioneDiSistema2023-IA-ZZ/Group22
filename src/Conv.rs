use ndarray::{Array1, Array4, arr1, Shape, Dim};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};
use std::cmp::max;
use ndarray::ArrayD;

#[derive(Clone, Debug)]
pub struct Conv{
    pub autopad: String,
    pub dilations: Array1<i32>,
    pub group: u32,
    pub kernel_shape: Shape<Dim<[usize; 2]>>,
    pub pads: Array1<i32>,
    pub strides: Array1<i32>,
}

impl Conv {
    pub fn new(ap: Option<String>,
               dil: Option<ndarray::Array1<i32>>,
               group: Option<u32>,
               kernel_shape: Option<Shape<Dim<[usize; 2]>>>,
               pads: Option<ndarray::Array1<i32>>,
               strides: Option<ndarray::Array1<i32>>, ) -> Conv {
        return Conv {
            autopad: ap.unwrap_or("NOT_SET".to_string()),
            dilations: dil.unwrap_or(arr1(&[1, 1])),
            group: group.unwrap_or(1),
            kernel_shape: kernel_shape.unwrap_or(Shape::from(Dim([1, 1]))),
            pads: pads.unwrap_or(arr1(&[0, 0, 0, 0])),
            strides: strides.unwrap_or(arr1(&[1, 1]))
        }
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Conv {

        let mut conv_tmp = Conv::new(None, None, None,
                                     None, None, None);
        for attr in attributes.iter() {
            match attr.name.as_str() {
                "autopad" => {
                    let string_result: Result<String, _> = String::from_utf8(attr.s.clone());

                    // Check if the conversion was successful
                    match string_result {
                        Ok(string) => {
                            conv_tmp.autopad = string;
                        }
                        Err(e) => {
                            println!("Error decoding Vec<u8>: {:?}", e);
                        }
                    }
                },
                "dilations" => {
                    // Step 1: Convert each element from i64 to i32
                    let converted_attr: Vec<i32> = attr.ints.clone().into_iter().map(|x| x as i32).collect();
                    // Step 2: Create an Array1<i32> from the converted Vec<i32>
                    let array_attr: Array1<i32> = arr1(&converted_attr);
                    conv_tmp.dilations = array_attr;
                },
                "group" => {
                    let converted_attr: u32 = attr.i.clone() as u32;
                    conv_tmp.group = converted_attr;
                },
                "kernel_shape" => {
                    //Convert from Vec<i64> to Shape<Dim<[usize; 2]>>
                    let mut kernel_vec: [usize; 2] = [0; 2];
                    let input = attr.ints.clone().into_iter().map(|val| val as usize).collect::<Vec<usize>>();
                    kernel_vec.copy_from_slice(&input);
                    conv_tmp.kernel_shape = Shape::from(Dim(kernel_vec));
                },
                "pads" => {
                    // Step 1: Convert each element from i64 to i32
                    let converted_attr: Vec<i32> = attr.ints.clone().into_iter().map(|x| x as i32).collect();
                    // Step 2: Create an Array1<i32> from the converted Vec<i32>
                    let array_attr: Array1<i32> = arr1(&converted_attr);
                    conv_tmp.pads = array_attr;
                },
                "strides" => {
                    // Step 1: Convert each element from i64 to i32
                    let converted_attr: Vec<i32> = attr.ints.clone().into_iter().map(|x| x as i32).collect();
                    // Step 2: Create an Array1<i32> from the converted Vec<i32>
                    let array_attr: Array1<i32> = arr1(&converted_attr);
                    conv_tmp.strides = array_attr;
                },

                _ => panic!("Attribute name not known")
            }
        }
        return conv_tmp;
    }
}


impl Compute for Conv{

    fn compute(&mut self, inputs: Input) -> Output {
        //return Output::Tensor32(Array4::from_elem((64,3,256,256), 1.5));

        let autopad = self.autopad.clone();
        let dilations = self.dilations.clone();
        let group = self.group.clone();
        let kernel_shape = self.kernel_shape.clone();
        let pads = self.pads.clone();
        let strides = self.strides.clone();

        let mut vec = match inputs {
            Input::Tensor4List(vec_array) => vec_array,
            _ => panic!("Input is not a vector")
        };
        // let mut x = vec[0].clone();
        let mut x1 = &vec[0];
        let mut x: Array4<f32> = x1.clone().into_dimensionality().unwrap();
        let mut w1 = vec[1].clone();
        //let mut b = vec[2].clone();

        // Retrieve input dimensions
        //let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);

        let (mut n, mut c, mut h, mut w) = (0, 0, 0, 0);

            // Get the size of each dimension
            let shape = x.shape();
            match shape.len() {
                4 => {
                    n = shape[0].clone();
                    c = shape[1].clone();
                    h = shape[2].clone();
                    w = shape[3].clone();
                },
                _ => panic!("Unexpected number of dimensions in the tensor"),
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
        // Calculate output dimensions based on autopad

/*
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


 */


        let oh = match autopad.as_str() {
            "SAME_UPPER" | "SAME_LOWER" => {
                let padding_h = if autopad == "SAME_UPPER" {
                    ((kh - 1) * dilations[0] as usize) / 2
                } else {
                    ((kh - 1) * dilations[0] as usize + 1) / 2
                };
                println!("padding_h: {}", padding_h);
                ((h + padding_h * 2 - kh * dilations[0] as usize + strides[0] as usize - 1) / strides[0] as usize) + 1
            }
            "VALID" | "NOT_SET" => {
                if pads.eq(&arr1(&[0, 0, 0, 0])) {
                    ((h - kh + 2 * pads[0] as usize) / strides[0] as usize) + 1
                    //((h - kh * dilations[0] as usize + strides[0] as usize - 1) / strides[0] as usize) + 1
                }else{
                    ((h + pads[0] as usize + pads[1] as usize + strides[0] as usize - kh * dilations[0] as usize - 1) / strides[0] as usize) + 1
                }
            },
            _ => {
                panic!("Invalid autopad mode")
            },
        };
        //println!("oh: {}", oh);

        let ow = match autopad.as_str() {
            "SAME_UPPER" | "SAME_LOWER" => {
                let padding_w = if autopad == "SAME_UPPER" {
                    ((kw - 1) * dilations[1] as usize) / 2
                } else {
                    println!("io1");
                    ((kw - 1) * dilations[1] as usize + 1) / 2
                };
                println!("padding_w: {}", padding_w);
                ((w + padding_w * 2 - kw * dilations[1] as usize + strides[1] as usize - 1) / strides[1] as usize) + 1
            }
            "VALID" | "NOT_SET" => {
                if pads.eq(&arr1(&[0, 0, 0, 0])) {

                    //((w - kw * dilations[1] as usize + strides[1] as usize - 1) / strides[1] as usize) + 1
                    ((w - kw + 2 * pads[0] as usize) / strides[0] as usize) + 1
                }else{
                    ((w + pads[2] as usize + pads[3] as usize + strides[1] as usize - kw * dilations[1] as usize - 1) / strides[1] as usize) + 1
                }
            },
            _ => panic!("Invalid autopad mode")
        };


        // Initialize output tensor
        let mut y = Array4::<f32>::zeros((n, m, oh, ow));

        // Convolution computation
        for ni in 0..n {
            for mi in 0..m {
                for hi in 0..oh {
                    for wi in 0..ow {
                        let mut sum = 0.0;

                        // Iterate over input channels, kernel height, and kernel width
                        for ci in 0..c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    // Calculate input indices with respect to padding and strides
                                    /*
                                    let mut xi= 0;
                                    let mut xj= 0;
                                    if ki == 0 && kj == 0 {
                                        xi = hi * strides[0] as usize + ki;
                                        xj = wi * strides[1] as usize + kj;
                                    }
                                    else if kj == 0 {
                                        xi = hi * strides[0] as usize + ki - pads[0] as usize;
                                        xj = wi * strides[1] as usize + kj;

                                    }else if ki == 0 {
                                        xi = hi * strides[0] as usize + ki;
                                        xj = wi * strides[1] as usize + kj - pads[2] as usize;
                                    }
                                    else {*/
                                        let xi = hi * strides[0] as usize + ki - pads[0] as usize;
                                        let xj = wi * strides[1] as usize + kj - pads[2] as usize;
                                    //}
                                    // Check if input indices are within bounds
                                    if xi < h as usize && xj < w as usize {
                                        // Access individual ArrayD within the Vec
                                        //let x_tensor: &ArrayD<f32> = &x[ni];
                                        // Access individual elements in input and weight tensors
                                        let w1_tensor = &w1;

                                        sum += x[[ni, ci, xi, xj]] * w1_tensor[[mi, ci, ki, kj]];
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        //TODO: check whether b exists or is None
                           // sum += b[mi];

                        // Set the computed sum in the output tensor
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

