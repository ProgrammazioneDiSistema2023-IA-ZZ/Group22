use ndarray::{Array4};
use crate::onnx_runtime::onnxruntime::{parse_input_tensor};
use crate::operations::{Input, Output};

mod onnx_proto3;
mod node;
mod add;
mod operations;
mod graph;
mod reshape;
mod soft_max;
mod dropout;
mod gemm;
mod concat;
mod maxpool;
mod start;
mod averagepool;
mod local_response_normalization;
mod relu;
mod matmul;
mod conv;
mod input;
mod onnx_runtime;


fn main() {
    let mut dep_graph = onnx_runtime::onnxruntime::get_computational_graph("src/mnist-7/model.onnx".to_string());
    let arr = parse_input_tensor("src/mnist-7/test_data_set_0/input_0.pb".to_string()).unwrap();
    let out = dep_graph.run(arr).unwrap();
    println!("Result from graph: ");
    let graph_result = match out {
        Output::TensorD(array) => array,
        _ => panic!("Errore output")
    };
    println!("{}", graph_result.clone());

    println!("Test_Data_set/Output_data: ");
    let arr = parse_input_tensor("src/mnist-7/test_data_set_0/output_0.pb".to_string()).unwrap();
    let result = match arr {
        Input::TensorD(array) => array,
        _ => panic!("Errore output")
    };
    println!("{}", result.clone());
    println!("Difference: ");
    let diff = graph_result - result;
    println!("{}", diff);

    let tmp_array: Vec<f32> = Array4::from_elem((1,1,28,28), 0.7).into_raw_vec();
    let net_input = Input::from_raw_vec(tmp_array, &[1, 1, 28, 28]).unwrap();
    let out = dep_graph.run(net_input).unwrap();
    let result = out.into_raw_vec().unwrap();
    result.into_iter().for_each(|val| print!("{} ", val));

    let mut dep_graph = onnx_runtime::onnxruntime::get_computational_graph("src/googlenet/model.onnx".to_string());
    let arr = parse_input_tensor("src/googlenet/test_data_set_0/input_0.pb".to_string()).unwrap();
    let out = dep_graph.run(arr).unwrap();
    println!("Result from graph: ");
    let graph_result = match out {
        Output::TensorD(array) => array,
        _ => panic!("Errore output")
    };
    println!("{}", graph_result.clone());

    println!("Test_Data_set/Output_data: ");
    let arr = parse_input_tensor("src/googlenet/test_data_set_0/output_0.pb".to_string()).unwrap();
    let result = match arr {
        Input::TensorD(array) => array,
        _ => panic!("Errore output")
    };
    println!("{}", result.clone());
    println!("Difference: ");
    let diff = graph_result - result;
    println!("{}", diff);
}

