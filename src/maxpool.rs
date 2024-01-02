use ndarray::{Array1, Array4, Array, arr1, Shape, Dim, par_azip, s, Axis};
use crate::operations::{Compute, Input, Output};
use crate::onnx_proto3::{AttributeProto, NodeProto};

#[derive(Clone, Debug)]
pub struct MaxPool{
    autopad: String,
    ceil_mode: u32,
    dilations: Array1<i32>,
    kernel_shape: Shape<Dim<[usize; 2]>>,
    pads: Array1<i32>,
    storage_order: u32,
    strides:  Array1<i32>
}

impl MaxPool{
    pub fn new(ap: Option<String>,
               cm: Option<u32>,
               dil: Option<ndarray::Array1<i32>>,
               kernel_shape: Shape<Dim<[usize; 2]>>,
               pads: Option<ndarray::Array1<i32>>,
               st_or: Option<u32>,
               strides: Option<ndarray::Array1<i32>>,
               ) -> MaxPool{
        return MaxPool{
            autopad: ap.unwrap_or("NOT_SET".to_string()),
            ceil_mode: cm.unwrap_or(0),
            dilations: dil.unwrap_or(arr1(&[1, 1])),
            kernel_shape: kernel_shape,
            pads: pads.unwrap_or(arr1(&[0, 0, 0, 0])),
            storage_order: st_or.unwrap_or(0),
            strides: strides.unwrap_or(arr1(&[1, 1]))
        }
    }

    pub fn parse_from_proto_node(attributes: &[AttributeProto]) -> Option<MaxPool>{
        //TODO Implement the method to parse from a vector of attributes
        return None;
    }
}

impl Compute for MaxPool {
    fn compute(&mut self, input: Input) -> Output {
        match input {
            Input::Tensor4(tensor) => {
                // Estrarre le dimensioni del tensore di input
                let (batch_size, channels, height, width) = tensor.dim();

                // Accedere direttamente ai valori delle dimensioni del kernel
                let (kernel_h, kernel_w) = (self.kernel_shape[0], self.kernel_shape[1]);

                // Utilizzare i valori dei riferimenti per strides e pads
                let stride_h = self.strides[0] as usize;
                let stride_w = self.strides[1] as usize;
                let pad_h = self.pads[0] as usize;
                let pad_w = self.pads[1] as usize;

                // Calcolare le dimensioni dell'output
                let output_height = ((height + 2 * pad_h - kernel_h) as f32 / stride_h as f32).ceil() as usize;
                let output_width = ((width + 2 * pad_w - kernel_w) as f32 / stride_w as f32).ceil() as usize;

                let mut output = Array4::zeros((batch_size, channels, output_height, output_width));

                for b in 0..batch_size {
                    for c in 0..channels {
                        for h in 0..output_height {
                            for w in 0..output_width {
                                let start_row = h * stride_h - pad_h;
                                let start_col = w * stride_w - pad_w;

                                let end_row = (start_row + kernel_h).min(height);
                                let end_col = (start_col + kernel_w).min(width);

                                let start_row = start_row.max(0) as usize;
                                let start_col = start_col.max(0) as usize;

                                // Estrarre la finestra corrente
                                let window = tensor.slice(s![b, c, start_row..end_row, start_col..end_col]);

                                // Trovare il valore massimo nella finestra
                                let max_value = window.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                                // Assegnare il valore massimo all'output
                                output[[b, c, h, w]] = max_value;
                            }
                        }
                    }
                }

                Output::Tensor4(output)
            },
            _ => panic!("Tipo di input errato per MaxPool"),
        }
    }
}
