use ndarray::{Array1, Array4, arr1, Shape, Dim, Dimension, s, Axis, IntoDimension, IxDyn, ShapeBuilder, Array};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};
use std::cmp::max;
use std::ops::Div;
use ndarray::ArrayD;
use ndarray::parallel::prelude::*;

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
                "auto_pad" => {
                    let string_result: Result<String, _> = String::from_utf8(attr.s.clone());

                    // Check if the conversion was successful
                    match string_result {
                        Ok(string) => {
                            conv_tmp.autopad = string;
                        }
                        Err(e) => {
                            panic!("Error decoding Vec<u8>: {:?}", e);
                        }
                    }
                },
                "autopad" => {
                    let string_result: Result<String, _> = String::from_utf8(attr.s.clone());

                    // Check if the conversion was successful
                    match string_result {
                        Ok(string) => {
                            conv_tmp.autopad = string;
                        }
                        Err(e) => {
                            panic!("Error decoding Vec<u8>: {:?}", e);
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

        /*let autopad = self.autopad.clone();
        let mut list = match inputs {
            Input::Tensor4List(array) => array,
            _ => panic!("wrong input type")
        };
        let w = list.pop().unwrap();
        let mut array = list.pop().unwrap();
        let shape = array.shape();
        if shape[1] == 1{
            //return Output::TensorD(Array4::from_elem((1,8,28,28), 1.2).into_shape(IxDyn(&[1,8,28,28])).unwrap());
            return Output::TensorD(get_first_conv());
        }else{
            //return Output::TensorD(Array4::from_elem((1,16,14,14), 23.2).into_shape(IxDyn(&[1, 16, 14, 14])).unwrap());
            return Output::TensorD(get_output2());
        }*/
        /*
        slice = np.ones((1, 3, 7, 7)) * 1
    w = np.ones((64, 3, 7, 7)) * 3
    b = np.ones(64)
    res = slice * w
    res = res.sum(axis=(1, 2, 3)) + b
        */
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
        let mut w2 = &vec[1];
        let mut w1 = w2.clone();

        // Retrieve input dimensions
        //let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);

        let (mut b, mut c, mut h, mut w) = (0, 0, 0, 0);

        // Get the size of each dimension
        let shape = x.shape();
        match shape.len() {
            4 => {
                b = shape[0].clone();
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

        let mut bias = Array1::<f32>::zeros((m));

        match vec.get(2) {
            Some(element) => bias = element.clone().into_shape((m,)).unwrap(),
            _=> bias = bias,
        };


        // Padding
        let mut left_h = pads[0] as usize;
        let mut left_w = pads[1] as usize;
        let mut right_h = pads[2] as usize;
        let mut right_w = pads[3] as usize;
        let kernel_size = kernel_shape.raw_dim().last_elem();
        let stride_h = strides[0] as usize;
        let stride_w = strides[1] as usize;

        // Calculate output dimensions based on autopad

        match autopad.as_str() {
            "SAME_UPPER" | "SAME_LOWER" => {
                //let oh = (((h + left_h + right_h - dilations[1] as usize * (kernel_size))/stride_h) + 1);
                //let width_padding_difference = w - kernel_size;
                let width_padding_difference = kernel_size/stride_w - 1;
                //I'd get the same value with height_padding_difference
                if width_padding_difference % 2 == 0 {
                    left_h = width_padding_difference.clone().div(2);
                    right_h = width_padding_difference.clone().div(2);
                    left_w = width_padding_difference.clone().div(2);
                    right_w = width_padding_difference.clone().div(2);
                }else{
                    if(autopad == "SAME_LOWER") {
                        left_h = width_padding_difference.clone().div(2) + 1;
                        right_h = width_padding_difference.clone().div(2);
                        left_w = width_padding_difference.clone().div(2) + 1;
                        right_w = width_padding_difference.clone().div(2);
                    }else {
                        left_h = width_padding_difference.clone().div(2);
                        right_h = width_padding_difference.clone().div(2) + 1;
                        left_w = width_padding_difference.clone().div(2);
                        right_w = width_padding_difference.clone().div(2) + 1;
                    }
                }
            }
            "VALID" | "NOT_SET" => {

            },
            _ => panic!("Invalid autopad mode")
        };

        let oh = (((h + left_h + right_h - dilations[1] as usize * (kernel_size))/stride_h) + 1);
        let ow = (((w + left_w + right_w - dilations[1] as usize * (kernel_size))/stride_w) + 1);
        //println!("{} - {} - {} - {}", oh, ow, stride_h, stride_w);
        //Create padded image

        //create an image by taking into account the padding; this is the padded input, not the output
        let mut padded_image = Array4::<f32>::zeros((b, c, h + left_h + right_h, w + left_w + right_w));
        //generate a mutable copy of x
        let mut original_view = x.view_mut();
        //generate a view on padded_image by only considering the pixels without padding
        let mut padded_view = padded_image.slice_mut(s![.., .., left_h..left_h + h, left_w..left_w + w]);
        //now x is the original image + the padded values
        padded_view.assign(&original_view);
        x = padded_image;


        // Initialize output tensor
        let mut y = Array4::<f32>::zeros((b, m, oh, ow));

        //println!("stride h: {}", stride_h);
        //println!("kernel size: {}", kernel_size);
        //println!("stride w: {}", stride_w);
        //println!("x: {}", x);

        let w_arr: Array4<f32> = w1.into_dimensionality().unwrap();
        for batch in (0..b) {
            for h in (0..oh) {
                for w in (0..ow){
                    let input_slice = x.slice(s![
                        batch,
                        ..,
                        h * stride_h..h * stride_h + kernel_size,
                        w * stride_w..w * stride_w + kernel_size
                    ]);

                    // Element-wise multiplication and sum
                    //println!("{}", input_slice);
                    //println!("{}", w1);

                    //println!("h: {}", h);
                    //println!("w: {}", w);
                    //println!("input slice: {}", input_slice);
                    //let mut convolution_result = &input_slice * &w1;
                    let mut results = Vec::new();
                    w_arr.clone()
                        .axis_iter(Axis(0))
                        .into_par_iter()
                        .map(|v| (&v * &input_slice).sum())
                        .collect_into_vec(&mut results);

                    let mut convolution_result= Array1::from(results);
                    //convolution_result = convolution_result.sum_axis(Axis(1)).sum_axis(Axis(1)).sum_axis(Axis(1));
                    //println!("conv result2: {}", convolution_result);
                    convolution_result = convolution_result + bias.clone();
                    //println!("{}", bias);

                    // Assign the result to the output array
                    //y[[batch, .., h, w]] = convolution_result;
                    let mut slice_y = y.slice_mut(s![batch, .., h, w]);
                    slice_y.assign(&convolution_result);
                    //y = y_temp;

                }
            }
        }
        /*println!("Shape :");
        y.shape().iter().for_each(|x| print!("{} ", x));*/
        Output::TensorD(y.into_dyn())
    }

    fn op_type(&self) -> &'static str {
        return "Conv";
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
