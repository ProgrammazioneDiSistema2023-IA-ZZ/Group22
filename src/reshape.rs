use ndarray::{Array1, Array4, arr1, Shape, Dim, Array2, Array3, Ix1, Ix2, Ix3, Ix4};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Reshape{
    pub shape: Vec<usize>,
}

impl Reshape{

    pub fn new(shape: Vec<usize>) -> Reshape{
        return Reshape{
            shape
        }
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Reshape{ //Change from Option to pure Conv
        let shape: Vec<usize> = attributes[0].ints.iter().map(|val| val.clone() as usize).collect();
        return Reshape{shape};
    }

}


impl Compute for Reshape{

    fn compute(&mut self, inputs: Input) -> Output {
        let reshaped =  match inputs {
            Input::Tensor32(array) => array.into_shape(self.shape.clone()).unwrap(),
            Input::Tensor1(array) => array.into_shape(self.shape.clone()).unwrap(),
            Input::Tensor2(array) => array.into_shape(self.shape.clone()).unwrap(),
            Input::Tensor3(array) => array.into_shape(self.shape.clone()).unwrap(),
            _ => panic!("Wrong input reshape")
        };

        return match reshaped.shape().len() {
            1 => Output::Tensor1(reshaped.into_dimensionality::<Ix1>().unwrap()),
            2 => Output::Tensor2(reshaped.into_dimensionality::<Ix2>().unwrap()),
            3 => Output::Tensor3(reshaped.into_dimensionality::<Ix3>().unwrap()),
            4 => Output::Tensor32(reshaped.into_dimensionality::<Ix4>().unwrap()),
            _ => panic!("Wrong shape dim reshape")
        }
    }
}

