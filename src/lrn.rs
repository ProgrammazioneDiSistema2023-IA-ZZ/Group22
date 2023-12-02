use ndarray::{Array1, Array4, arr1, Shape, Dim, Array2, Array3, ArrayD, Ix1, Ix2, Ix3, Ix4, Axis, IndexLonger};
use ndarray::parallel::prelude::{ParallelIterator, IntoParallelIterator};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};
use crate::operations::Output::Tensor32;

#[derive(Clone, Debug)]
pub struct LRN{
    pub alpha: f32,
    pub beta: f32,
    pub bias: f32,
    pub size: usize

}

impl LRN{
    pub fn new(al: Option<f32>,
               bt: Option<f32>,
               bs: Option<f32>,
               sz: usize) -> LRN{
        return LRN{
            alpha: al.unwrap_or(0.0001),
            beta: bt.unwrap_or(0.75),
            bias: bs.unwrap_or(1.0),
            size: sz
        }

    }
    /*
    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> LRN{ //Change from Option to pure Conv
        let shape: Vec<usize> = attributes[0].ints.iter().map(|val| val.clone() as usize).collect();
        return LRN{shape};
    }
    */
}


impl Compute for LRN {
    fn compute(&mut self, inputs: Input) -> Output {
        match inputs {
            Input::Tensor32(input) => {
                let shape = input.shape();
                let half_size = self.size.clone() / 2;
                let alpha_div_size = self.alpha.clone() / self.size.clone() as f32;

                // Specifica esplicitamente le dimensioni dell'array output come [N, C, D1, D2]
                let mut output = Array4::zeros((shape[0].clone(), shape[1].clone(), shape[2].clone(), shape[3].clone()));

                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut subview| {
                        for mut row in subview.axis_iter_mut(Axis(0)) {
                            for mut col in row.axis_iter_mut(Axis(0)) {
                                for mut depth in col.axis_iter_mut(Axis(0)) {
                                    let c = depth.
                                        index();
                                    let start_i = if c >= half_size {
                                        c - half_size.clone()
                                    } else {
                                        0
                                    };
                                    let end_i = if c + half_size.clone() < shape[1] {
                                        c + half_size.clone()
                                    } else {
                                        shape[1].clone() - 1
                                    };

                                    let square_sum: f32 = depth
                                        .index_axis(Axis(0), start_i..=end_i)
                                        .map(|&val| val * val)
                                        .sum();

                                    for normalization in depth.iter_mut() {
                                        *normalization =
                                            *normalization.cloned() / (self.bias.clone() + alpha_div_size.clone() * square_sum.clone()).powf(self.beta.clone());
                                    }
                                }
                            }
                        }
                    });

                Output::Tensor32(output)
            }
            _ => panic!("Wrong input"),
        }
    }
}
