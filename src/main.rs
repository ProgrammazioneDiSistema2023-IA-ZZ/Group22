extern crate core;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use crate::onnx_proto3::{AttributeProto, ModelProto, NodeProto, TypeProto_oneof_value};
use protobuf::{Message, ProtobufEnum};
use ndarray::{arr1, Array1, Array2, Array4, ArrayD, Dim, Ix2, Ix4, IxDyn, Shape};
use crate::add::{Add, AddToTryGraph};
use crate::gemm::Gemm;
use crate::graph::DepGraph;
use crate::maxpool::MaxPool;
use crate::node::{Node, SimpleNode};
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

#[cfg(test)]
mod tests {
    use std::cmp::max;
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::Read;
    use ndarray::{arr1, Array, Array3, Dim, Dimension, Ix4, Shape};
    use ndarray::{Array1, Array2, Array4, ArrayD, Ix2, IxDyn};
    use protobuf::Message;
    use crate::add::Add;
    use crate::averagepool::AveragePool;
    use crate::gemm::Gemm;
    use crate::local_response_normalization::LRN;
    use crate::maxpool::MaxPool;
    use crate::onnx_proto3::{ModelProto, NodeProto};
    use crate::operations::{Compute, Input, Output};

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn test_max_pool_stride1(){
        let mut max_pool_node = MaxPool::new(Some(Shape::from(Dim([7, 7]))),
                                             Some(arr1(&[0,0,0,0])), Some(arr1(&[1,1])));
        let mut prova = Array4::from_elem((1, 3, 9, 9), 0.0);
        let mut comparison = Array4::from_elem((1, 3, 3, 3), 0.0);
        for i in 0..3 {
            prova[[0, i, 0, 0]] = 1.0;
            prova[[0, i, 0, 8]] = 1.0;
            prova[[0, i, 8, 0]] = 1.0;
            prova[[0, i, 8, 8]] = 1.0;
            comparison[[0, i, 0, 0]] = 1.0;
            comparison[[0, i, 0, 2]] = 1.0;
            comparison[[0, i, 2, 0]] = 1.0;
            comparison[[0, i, 2, 2]] = 1.0;
        }
        let input_d = Input::TensorD(prova.into_shape(IxDyn(&[1, 3, 9, 9])).unwrap());
        let result = match max_pool_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, comparison);
    }

    #[test]
    fn test_max_pool_stride2(){
        let mut kernel_vec:[usize; 2] = [0; 2];
        let mut input: Vec<i64> = vec![7, 7];
        let input = input.into_iter().map(|val| val as usize).collect::<Vec<usize>>();
        kernel_vec.copy_from_slice(&input);
        let mut max_pool_node = MaxPool::new(Some(Shape::from(Dim(kernel_vec) )),
                                             Some(arr1(&[0,0,0,0])), Some(arr1(&[2,2])));
        let mut prova = Array4::from_elem((1, 3, 9, 9), 0.0);
        let mut comparison = Array4::from_elem((1, 3, 2, 2), 0.0);
        for i in 0..3 {
            prova[[0, i, 0, 0]] = 1.0;
            prova[[0, i, 0, 8]] = 2.0;
            prova[[0, i, 8, 0]] = 1.5;
            prova[[0, i, 8, 8]] = 3.0;
            prova[[0, i, 4, 4]] = 0.5;
            comparison[[0, i, 0, 0]] = 1.0;
            comparison[[0, i, 0, 1]] = 2.0;
            comparison[[0, i, 1, 0]] = 1.5;
            comparison[[0, i, 1, 1]] = 3.0;
        }
        let input_d = Input::TensorD(prova.into_shape(IxDyn(&[1, 3, 9, 9])).unwrap());
        let result = match max_pool_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, comparison);
    }

    #[test]
    fn test_average_pool_stride2(){
        let mut avg_pool_node = AveragePool::new(Some(Shape::from(Dim([7, 7]))),
                                             Some(arr1(&[0,0,0,0])), Some(arr1(&[2,2])));
        let mut prova = Array4::from_elem((1, 3, 9, 9), 1.0);
        let mut comparison = Array4::from_elem((1, 3, 2, 2), 1.0);
        let input_d = Input::TensorD(prova.into_shape(IxDyn(&[1, 3, 9, 9])).unwrap());
        let result = match avg_pool_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, comparison);
    }

    #[test]
    fn test_lrn(){
        // Create two test Array4 instances (representing images)
        let vec1 = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        ];
        let vec2 = vec![0.99795485,  1.9948157,   2.9902866,   3.9840744,   4.9758863,   5.965434,
              6.95243,     7.936592,    8.917418,    9.894319,   10.86688,    11.834699,
             12.797375,   13.754522,   14.705759,   15.650717,   16.844427,   17.81153,
             18.774221,   19.732246,   20.685343,   21.633266,   22.57577,    23.512613
        ];
        let test_data_1: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 3, 2, 4])), vec1).unwrap();

        let test_data_2: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 3, 2, 4])), vec2).unwrap();

        let mut lrn_node = LRN::new(0.0001, 0.75, 1.0, 3);

        let input_d = Input::TensorD(test_data_1.into_shape(IxDyn(&[1, 3, 2, 4])).unwrap());
        let result = match lrn_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, test_data_2);
    }

    #[test]
    fn test_max_pool_from_python_results(){
        let vec1 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0
        ];
        let vec2 = vec![
            13.0, 14.0, 15.0, 15.0, 15.0,
            18.0, 19.0, 20.0, 20.0, 20.0,
            23.0, 24.0, 25.0, 25.0, 25.0,
            23.0, 24.0, 25.0, 25.0, 25.0,
            23.0, 24.0, 25.0, 25.0, 25.0
        ];
        let test_data_1: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 1, 5, 5])), vec1).unwrap();

        let test_data_2: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 1, 5, 5])), vec2).unwrap();

        let mut max_pool_node = MaxPool::new(Some(Shape::from(Dim([5, 5]) )),
                                             Some(arr1(&[2,2,2,2])), Some(arr1(&[1,1])));

        let input_d = Input::TensorD(test_data_1.into_shape(IxDyn(&[1, 1, 5, 5])).unwrap());
        let result = match max_pool_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, test_data_2);
    }

    #[test]
    fn test_max_pool_parsing(){
        let mut input_onnx = File::open("src/gender_googlenet.onnx").unwrap();
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
        let graph = model.get_graph();
        //Estrazione dei nodi dal protoGrafo
        let nodes = graph.get_node();
        let mut max_nodes: Vec<MaxPool> = Vec::new();
        let mut count = 0;

        for node in nodes.iter(){
            if node.op_type == "MaxPool"{
                count+=1;
                max_nodes.push(MaxPool::parse_from_proto_node(node.attribute.as_slice()));
            }
        }
        for node in max_nodes.iter(){
            print!("Kernel Shape: ");
            node.kernel_shape.raw_dim().slice().iter().for_each(|el| print!("{}", el));
            println!();
            print!("Pads: ");
            node.pads.iter().for_each(|el| print!("{}", *el));
            println!();
            print!("Strides: ");
            node.strides.iter().for_each(|el| print!("{}", *el));
            println!();
            println!();
        }
        assert_eq!(max_nodes.len(), count);
    }

    #[test]
    fn test_lrn_parsing(){
        let mut input_onnx = File::open("src/gender_googlenet.onnx").unwrap();
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
        let graph = model.get_graph();
        //Estrazione dei nodi dal protoGrafo
        let nodes = graph.get_node();
        let mut lrn_nodes: Vec<LRN> = Vec::new();
        let mut count = 0;

        for node in nodes.iter(){
            if node.op_type == "LRN"{
                count+=1;
                lrn_nodes.push(LRN::parse_from_proto_node(node.attribute.as_slice()));
            }
        }
        for node in lrn_nodes.iter(){
            println!("{} - {} - {} - {}", node.alpha, node.beta, node.bias, node.size);
        }
        assert_eq!(lrn_nodes.len(), count);
    }

    #[test]
    fn test_gemm_parsing(){
        let mut input_onnx = File::open("src/gender_googlenet.onnx").unwrap();
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
        let graph = model.get_graph();
        //Estrazione dei nodi dal protoGrafo
        let nodes = graph.get_node();
        let mut gemm_nodes: Vec<Gemm> = Vec::new();
        let mut count = 0;

        for node in nodes.iter(){
            if node.op_type == "Gemm"{
                count+=1;
                gemm_nodes.push(Gemm::parse_from_proto_node(node.attribute.as_slice()));
            }
        }
        for node in gemm_nodes.iter(){
            println!("{} - {} - {} - {}", node.alpha, node.beta, node.trans_a, node.trans_b);
        }
        assert_eq!(gemm_nodes.len(), count);
    }

    #[test]
    fn test_add(){
        let vec1 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0
        ];
        let vec2 = vec![
            13.0, 14.0, 15.0, 15.0
        ];

        //Retrieved from python NumPy
        let vec3 = vec![14.0, 15.0, 16.0, 17.0, 19.0,
                        20.0, 21.0, 22.0, 24.0, 25.0, 26.0, 27.0,
                        28.0, 29.0, 30.0, 31.0];

        let to_compare = Array4::from_shape_vec(
            Shape::from(Dim([1, 4, 2, 2])), vec3).unwrap();

        let test_data_1: Array4<f32> = Array4::from_shape_vec(
            Shape::from(Dim([1, 4, 2, 2])), vec1).unwrap();

        let test_data_2: Array3<f32> = Array3::from_shape_vec(
            Shape::from(Dim([4, 1, 1])), vec2).unwrap();

        let mut add_node = Add::new();

        let input_1 = test_data_1.into_shape(IxDyn(&[1, 4, 2, 2])).unwrap();
        let input_2 = test_data_2.into_shape(IxDyn(&[4, 1, 1])).unwrap();
        let input_d = Input::Tensor4List(Vec::from([input_1, input_2]));
        let result = match add_node.compute(input_d) {
            Output::TensorD(arr) => arr.into_dimensionality::<Ix4>().unwrap(),
            _ => panic!("Wrong result")
        };
        assert_eq!(result, to_compare);
    }
}

fn main() {
    //Script per estrarre onnx_proto3.rs tramite protocol buffer
    /*protoc_rust::Codegen::new()
        .out_dir("src")
        .inputs(&["protos/onnx.proto3"])
        .include("protos")
        .run()
        .expect("protoc");*/

    //Lettura onnx file
    let mut input_onnx = File::open("src/gender_googlenet.onnx").unwrap();
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
    let start_node = Node::new(previous.to_string(),
                               Box::new(Start::new()));
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
    let mut node_reshape = Reshape{shape: vec![3, 4, 1]};
    let input = Input::TensorD(Array2::from_elem((3, 4), 1.3).into_shape(IxDyn(&[3, 4])).unwrap());
    let output = node_reshape.compute(input);
    if let Output::Tensor3(array) = output {
        println!("GODO FUNZIONA, MICHELE SEI UNA MERDA");
        println!("{}", array);
    }
    let reshape_node_parsed = Reshape::parse_from_proto_node(&reshape_node.unwrap().attribute);
    reshape_node_parsed.shape.into_iter().for_each(|val| print!("{} ", val));

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

}

