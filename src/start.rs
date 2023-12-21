use ndarray::Array4;
use crate::operations::{Compute, Input, Output};

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