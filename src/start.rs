use ndarray::{ArrayD};
use crate::operations::{Compute, Input, Output};

pub struct Start {

}

impl Start {
    pub fn new() -> Self {
        Start{}
    }
}

impl Compute for Start{

    fn compute(&mut self, inputs: Input) -> Output {
        return match inputs {
            Input::TensorD(vec) => Output::TensorD(vec),
            _ => panic!("Wrong starting input type")
        }
    }
}