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
    match out {
        Output::TensorD(array) => println!("{}", array),
        _ => println!("Errore output")
    }

    let mut arr = parse_input_tensor("src/mnist-7/test_data_set_0/input_0.pb".to_string()).unwrap();
    let out = dep_graph.run(arr).unwrap();
    match out {
        Output::TensorD(array) => println!("{}", array),
        _ => println!("Errore output")
    }

    let tmp_array: Vec<f32> = Array4::from_elem((1,1,28,28), 0.7).into_raw_vec();
    let net_input = Input::from_raw_vec(tmp_array, &[1, 1, 28, 28]).unwrap();
    let out = dep_graph.run(net_input).unwrap();
    let result = out.into_raw_vec().unwrap();
    result.into_iter().for_each(|val| print!("{} ", val));

    println!();
    let arr = parse_input_tensor("src/mnist-7/test_data_set_0/output_0.pb".to_string()).unwrap();
    let raw_out = arr.into_raw_vec().unwrap();
    raw_out.into_iter().for_each(|val| print!("{} ", val));


    return;

    //Script per estrarre onnx_proto3.rs tramite protocol buffer
    /*protoc_rust::Codegen::new()
        .out_dir("src")
        .inputs(&["protos/onnx.proto3"])
        .include("protos")
        .run()
        .expect("protoc");*/

    //Lettura onnx file
    let mut input_onnx = File::open("src/mnist-7.onnx").unwrap();
    //Onnx file into byte array
    let mut byte_array = Vec::<u8>::new();
    input_onnx.read_to_end(&mut byte_array).unwrap();
    //Parsing del byte array nella struttura onnx_proto3.rs
    let model: ModelProto = match Message::parse_from_bytes(&byte_array) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("Failed to parse the ONNX model: {}", err);
            return;
        }
    };
    //Estrazione grafo dal modello Proto
    let graph = model.get_graph();
    //How to transform a TensorProto into  Vec<f32>
    /*for val in graph.get_initializer().iter(){
        if val.get_name() == "loss3/classifier_w_0" {
            let mut raw = val.get_raw_data();
            println!("{}", raw.len());
            let floats: Vec<f32> = raw
                .chunks_exact(4) // Split into chunks of 4 bytes (size of f32)
                .map(|chunk| {
                    let mut bytes_array = [0; 4];
                    bytes_array.copy_from_slice(chunk);
                    f32::from_bits(u32::from_le_bytes(bytes_array)) // Convert u8 to f32
                })
                .collect();
            let mut i = 0;
            for el in floats.into_iter(){
                print!("{} ", el);
                i += 1;
                if i % 10 == 0 {println!()}
                if i == 1000 {return;}
            }
        }
    }
    return;*/

    //Estrazione dei nodi dal protoGrafo
    let nodes = graph.get_node();
    //Estrazione dei nomi delle operazioni con hash set per velocizzare sviluppo
    let mut class_map = HashSet::<String>::new();
    let mut reshape_node: Option<NodeProto> = None;

    for node in nodes.iter(){
        println!("name: {}", node.name.clone());
        println!("dom: {}", node.get_domain());
        println!("str: {}", node.get_doc_string());
        println!("op: {}", node.get_op_type());
        /*if node.op_type == "LRN"{
           for attr in node.attribute.iter(){
               print!("{} ", attr.name);
               print!("{} ", attr.field_type.value());
               print!("{}", attr.f);
               print!("{}", attr.i);
               println!();
           }
            node.get_input().iter().for_each(|s| println!("{}", s.clone()));
        }*/
        if node.op_type == "Reshape"{
            reshape_node = Some(node.clone());
        }
        class_map.insert(node.op_type.clone());
    }

    //stampa degli op_type di ogni operazione
    class_map.into_iter().for_each(|el| {println!("{}", el)});
    return;

}

/*
let mut gemm_node = Gemm::new(None, None, None, Some(1));
    let input_gemm = Array2::from_elem((1, 1024), 1.3).into_shape(IxDyn(&[1, 1024])).unwrap();
    let b_vec = Array2::from_elem((1000, 1024), 3.0).into_shape(IxDyn(&[1000, 1024])).unwrap();
    let c_vec = Array1::from_elem(1000, 2.0).into_shape(IxDyn(&[1000])).unwrap();
    let inputs = Input::Tensor4List(Vec::from([input_gemm, b_vec, c_vec]));
    let out = gemm_node.compute(inputs);

    //return;

    /*//EXAMPLE CONV NODE USAGE
    let mut conv_node = Conv::new(None, None, None, None, None, None, Array4::from_elem((64,3,256,256), 1.3));
    let first_input = Array4::from_elem((64,3,256,256), 1.3);
    let output = match conv_node.compute(Input::Tensor32(first_input)) {
        Output::Tensor32(vec) => vec,
        _ => panic!("wrong output")
    };
    println!("{}", output);*/

    let mut nodes = HashMap::<String, Node>::new();
    let mut previous = "Start";
    /*let start_node = Node::new(previous.to_string(),
                               Box::new(Start::new()));*/
    nodes.insert(start_node.id(), start_node);
    let x: u16 = 2;

    for (id, costant) in [("A", x), ("B",3), ("C",2), ("D", 1), ("X", 2), ("Y", 6)] {
        //let mut conv_node = Conv::new(None, None, None,
                                      //None, None, None,
                                      //Array4::from_elem((64, 3, 256, 256), 1.3));
        let mut add_node = AddToTryGraph::new(f32::from(costant));
        let mut node = Node::new(id.to_string(), Box::new(add_node));
        let previous = match id {
            "A" => "Start",
            "B" => "A",
            "C" => "A",
            "D" => "B",
            "X" => "A",
            "Y" => "X",
            _ => "Start"
        };
        node.add_dep(previous.to_string());
        if id == "D" {
            node.add_dep("C".to_string());
            node.add_dep("Y".to_string());
            node.add_dep("A".to_string());
        }
        if id == "X" {
            node.add_dep("B".to_string());
        }
        nodes.insert(id.to_string(), node);
    }
    let mut dep_graph = DepGraph::new(nodes);
    let final_result = match dep_graph.run().unwrap() {
        Output::Tensor32(vec) => vec,
        _ => panic!("wrong output")
    };
    println!("{}", final_result);

    /*for (key, deps) in dep_graph.deps.read().unwrap().iter(){
        println!("Node: {}", key.clone());
        print!("Deps:   ");
        for dep in deps.iter(){
            print!("{}", dep.clone());
        }
        println!();
    }*/
    /*for key in ["Start", "A", "B", "C", "D"]{
        let mut current = nodes.remove(key).unwrap();
        println!("Iterating on: {}", current.id().clone());
        current.compute_operation(&nodes);
        nodes.insert(key.to_string(), current);
        let next_nodes = graph::remove_node_id(key.to_string(),
                                                      &dep_graph.deps,
                                                      &dep_graph.rdeps).unwrap();
        next_nodes.into_iter().for_each(|node| dep_graph.ready_nodes.push(node));
    }
    let out = match nodes.remove("D").unwrap().output.unwrap(){
        Output::Tensor32(vec) => vec
    };
    print!("{}", out);*/
    let mut node_reshape = Reshape{};
    let input = Input::TensorD(Array2::from_elem((3, 4), 1.3).into_shape(IxDyn(&[3, 4])).unwrap());
    let output = node_reshape.compute(input);
    if let Output::Tensor3(array) = output {
        println!("GODO FUNZIONA, MICHELE SEI UNA MERDA");
        println!("{}", array);
    }
    let reshape_node_parsed = Reshape::parse_from_proto_node(&reshape_node.unwrap().attribute);
    //reshape_node_parsed.shape.into_iter().for_each(|val| print!("{} ", val));

    let input_vec:Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 10000.0, 10001.0, 10002.0, 10003.0];
    let good_result: Vec<f32> = vec![0.032058604, 0.08714432,  0.23688284,  0.6439143,
                                     0.032058604, 0.08714432,  0.23688284,  0.6439143  ];
    let input_d = Input::TensorD(ArrayD::from_shape_vec(IxDyn(&[2, 4]), input_vec).unwrap());
    let mut softmax_node = SoftMax::new();
    let result = match softmax_node.compute(input_d) {
        Output::TensorD(arr) => arr.into_dimensionality::<Ix2>().unwrap().into_raw_vec(),
        _ => panic!("Wrong result")
    };
    good_result.iter().for_each(|val| print!("{}", val));
    println!();
    result.iter().for_each(|val| print!("{}", val));
    assert_eq!(good_result, result);

*/

