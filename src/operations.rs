use std::fmt::Debug;
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, IxDyn};
use crate::onnx_runtime::onnxruntime::{Error};


pub trait Compute {
    fn compute(&mut self, inputs: Input) -> Output;
    fn op_type(&self) -> &'static str;
}
/*
impl Clone for Box<dyn Compute> {
    fn clone(&self) -> Self {
        self.clone()
    }
}

 */

impl Debug for dyn Compute {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dyn Compute")
    }
}

/*#[derive(Clone, Debug)]
pub enum Operation{
    Convolution(Conv),
    Addition(Add)
}

impl Operation {
    pub fn get_operation(op_type: &str, node: NodeProto) -> Option<Self>{
        match op_type {
            "Conv" => Some(Convolution(Conv::parse_from_proto_node(&node.get_attribute()).unwrap())),
            "Add" => Some(Addition(Add::parse_from_proto_node(&node.get_attribute()).unwrap())),
            _ => None
        }
    }
}*/

#[derive(Clone, Debug)]
pub enum Input {
    Tensor32(Array4<f32>),
    Tensor1(Array1<f32>),
    Tensor2(Array2<f32>),
    Tensor3(Array3<f32>),
    Tensor4(Array4<f32>),
    TensorD(ArrayD<f32>),
    Tensor32Vec(Vec<Array4<f32>>),
    Tensor4List(Vec<ArrayD<f32>>),
    Empty
}

#[derive(Clone, Debug)]
pub enum Output {
    Tensor32(Array4<f32>),
    Tensor1(Array1<f32>),
    Tensor2(Array2<f32>),
    Tensor3(Array3<f32>),
    Tensor4(Array4<f32>),
    TensorD(ArrayD<f32>)
}

impl Input {
    pub fn from_raw_vec(vec: Vec<f32>, shape: &[usize]) -> Result<Input, Error>{
        let res = match shape.len() {
            d if d >= 1 && d <= 4 => {
                ArrayD::from_shape_vec(IxDyn(shape), vec)
            },
            _ => return Err(Error::ShapeError)
        };
        return match res {
            Ok(val) => Ok(Input::TensorD(val)),
            Err(e) => Err(Error::ConversionError)
        }
    }

    pub fn into_raw_vec(self) -> Result<Vec<f32>, Error> {
        match self {
            Input::Tensor32(arr) => Ok(arr.into_raw_vec()),
            Input::Tensor1(arr) => Ok(arr.into_raw_vec()),
            Input::Tensor2(arr) => Ok(arr.into_raw_vec()),
            Input::Tensor3(arr) => Ok(arr.into_raw_vec()),
            Input::Tensor4(arr) => Ok(arr.into_raw_vec()),
            Input::TensorD(arr) => Ok(arr.into_raw_vec()),
            _ => Err(Error::ConversionError)
        }
    }

    pub fn list_into_raw_vec(self) -> Result<Vec<Vec<f32>>, Error> {
        match self {
            Input::Tensor4List(vec) =>
                Ok(vec.into_iter().map(|val| val.into_raw_vec()).collect::<Vec<Vec<f32>>>()),
            Input::Tensor32Vec(vec) =>
                Ok(vec.into_iter().map(|val| val.into_raw_vec()).collect::<Vec<Vec<f32>>>()),
            _ => Err(Error::ConversionError)
        }
    }

}

impl Output {
    pub fn from_raw_vec(vec: Vec<f32>, shape: &[usize]) -> Result<Output, Error>{
        let res = match shape.len() {
            d if d >= 1 && d <= 4 => {
                ArrayD::from_shape_vec(IxDyn(shape), vec)
            },
            _ => return Err(Error::ShapeError)
        };
        return match res {
            Ok(val) => Ok(Output::TensorD(val)),
            Err(e) => Err(Error::ConversionError)
        }
    }

    pub fn into_raw_vec(self) -> Result<Vec<f32>, Error> {
        match self {
            Output::Tensor32(arr) => Ok(arr.into_raw_vec()),
            Output::Tensor1(arr) => Ok(arr.into_raw_vec()),
            Output::Tensor2(arr) => Ok(arr.into_raw_vec()),
            Output::Tensor3(arr) => Ok(arr.into_raw_vec()),
            Output::Tensor4(arr) => Ok(arr.into_raw_vec()),
            Output::TensorD(arr) => Ok(arr.into_raw_vec()),
            //_ => Err(Error::ConversionError)
        }
    }

}

