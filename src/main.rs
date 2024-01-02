use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use crate::onnx_proto3::{AttributeProto, ModelProto, NodeProto, TensorProto, TypeProto_oneof_value};
use protobuf::{Message, ProtobufEnum};
use ndarray::{arr1, Array1, Array2, Array4, ArrayD, Dim, Ix2, Ix4, IxDyn, Shape};
use crate::add::{Add, AddToTryGraph};
use crate::gemm::Gemm;
use crate::graph::DepGraph;
use crate::maxpool::MaxPool;
use crate::node::{Node, SimpleNode};
use crate::onnx_runtime::onnxruntime::{get_nodes, parse_input_tensor};
use crate::operations::{Compute, Input, Output};
use crate::reshape::Reshape;
use crate::soft_max::SoftMax;
use crate::start::Start;

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
mod onnx_runtime;
mod Conv;
mod input;

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

