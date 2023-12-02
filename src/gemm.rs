use ndarray::{Array1, Array4, arr1, Shape, Dim, Array2, IxDyn};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct Gemm{
    alpha: f32,
    beta: f32,
    transA: i32,
    transB: i32,
    input_order: Vec<String>
}

impl Gemm{
    pub fn new(
        alpha: Option<f32>,
        beta: Option<f32>,
        trans_a: Option<i32>,
        trans_b: Option<i32>,
        input_order: Vec<String>
    ) -> Gemm{
        return Gemm{
            alpha: alpha.unwrap_or(1.0),
            beta: beta.unwrap_or(1.0),
            transA: trans_a.unwrap_or(0),
            transB: trans_b.unwrap_or(0),
            input_order: Vec::<String>::new()
        }

    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Option<Gemm>{ //Change from Option to pure Conv
        //TODO Implement the method to parse from a vector of attributes
        return None;
    }

}


impl Compute for Gemm{

    fn compute(&mut self, inputs: Input) -> Output {
        let mut arrays = match inputs {
            Input::Tensor4List(vec_array) => vec_array,
            _ => panic!("Input is not a vector")
        };
        let mut c = arrays.pop().unwrap();
        let mut b: Array2<f32> = arrays.pop().unwrap().into_dimensionality().unwrap();
        let mut a: Array2<f32>  = arrays.pop().unwrap().into_dimensionality().unwrap();
        if self.transA != 0 {
            a = a.reversed_axes();
        }
        if self.transB != 0 {
            b = b.reversed_axes();
        }
        let result  = self.alpha * a.dot(&b) + self.beta * c;
        let out_len  = Vec::from(result.shape());
        return Output::TensorD(result.into_shape(IxDyn(&out_len)).unwrap());
    }
}
