use ndarray::{Array1, Array4, arr1, Shape, Dim, Array2, Array3, Ix1, Ix2, Ix3, Ix4, IxDyn};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Reshape{
}

impl Reshape{

    pub fn new(shape: Vec<usize>) -> Reshape{
        return Reshape{
        }
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Reshape{ //Change from Option to pure Conv
        //let shape: Vec<usize> = attributes[0].ints.iter().map(|val| val.clone() as usize).collect();
        return Reshape{};
    }

}

impl Compute for Reshape{

    fn compute(&mut self, inputs: Input) -> Output {
        let mut list =  match inputs {
            Input::Tensor4List(array) => array,
            _ => panic!("Wrong input reshape")
        };
        let mut shape = list.pop().unwrap().map(|val| (*val) as usize).into_raw_vec();
        let mut vec = list.pop().unwrap();
        //println!("shape: {:?}", shape);
        //vec.shape().iter().for_each(|val| print!("{}",val));
        //println!();
        let reshaped = vec.into_shape(IxDyn(&shape)).unwrap();
        return Output::TensorD(reshaped);
    }

    fn op_type(&self) -> &'static str {
        return "Reshape";
    }
}