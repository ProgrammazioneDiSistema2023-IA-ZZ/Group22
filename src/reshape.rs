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
        /*
        OLD VERSION
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
    */
        let mut list =  match inputs {
            Input::Tensor4List(array) => array,
            _ => panic!("Wrong input reshape")
        };
        let mut shape = list.pop().unwrap().map(|val| (*val) as usize).into_raw_vec();
        let mut vec = list.pop().unwrap();
        let reshaped = vec.into_shape(IxDyn(&shape)).unwrap();
        return Output::TensorD(reshaped);
    }

    fn op_type(&self) -> &'static str {
        return "Reshape";
    }
}