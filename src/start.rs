use ndarray::{ArrayD};
use crate::operations::{Compute, Input, Output};

pub struct Start {
    data: Input
}

impl Start {
    pub fn new(input: ArrayD<f32>) -> Self {
        Start{data: Input::TensorD(input)}
    }
}

impl Compute for Start{

    fn compute(&mut self, inputs: Input) -> Output {
        return match self.data.clone() {
            Input::TensorD(vec) => Output::TensorD(vec),
            _ => panic!("Wrong starting input type")
        }
    }
}