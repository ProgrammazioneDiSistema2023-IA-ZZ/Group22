use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::Hash;
use std::io::Read;
use crate::onnx_proto3::ModelProto;
use protobuf::{Message};
use crate::conv::{Conv, Start};
use ndarray::Array4;
use crate::add::Add;
use crate::graph::DepGraph;
use crate::node::{Node, SimpleNode};
use crate::operations::{Compute, Input, Output};
mod conv;
mod onnx_proto3;
mod node;
mod add;
mod operations;
mod graph;

fn main() {
    //Script per estrarre onnx_proto3.rs tramite protocol buffer
    /*protoc_rust::Codegen::new()
        .out_dir("src")
        .inputs(&["protos/onnx.proto3"])
        .include("protos")
        .run()
        .expect("protoc");*/

    //Lettura onnx file
    let mut input_onnx = File::open("src/mobilenetv2-10.onnx").unwrap();
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
    //Estrazione dei nodi dal protoGrafo
    let nodes = graph.get_node();
    //Estrazione dei nomi delle operazioni con hash set per velocizzare sviluppo
    let mut class_map = HashSet::<String>::new();
    for node in nodes.iter(){
        class_map.insert(node.op_type.clone());
    }
    //stampa degli op_type di ogni operazione
    class_map.into_iter().for_each(|el| {println!("{}", el)});

    //EXAMPLE CONV NODE USAGE
    let mut conv_node = Conv::new(None, None, None, None, None, None, Array4::from_elem((64,3,256,256), 1.3));
    let first_input = Array4::from_elem((64,3,256,256), 1.3);
    let output = match conv_node.compute(Input::Tensor32(first_input)) {
        Output::Tensor32(vec) => vec
    };
    println!("{}", output);

    let mut nodes = HashMap::<String, Node>::new();
    let mut nodes_vec = Vec::<Node>::new();
    let mut previous = "Start";
    let start_node = Node::new(previous.to_string(),
                               Box::new(Start::new(Array4::from_elem((64, 3, 256, 256), 1.5))));
    nodes.insert(start_node.id(), start_node);
    let x: u16 = 2;

    for (id, costant) in [("A", x), ("B",3), ("C",2), ("D", 1)] {
        //let mut conv_node = Conv::new(None, None, None,
                                      //None, None, None,
                                      //Array4::from_elem((64, 3, 256, 256), 1.3));
        let mut add_node = Add::new(f32::from(costant));
        let mut node = Node::new(id.to_string(), Box::new(add_node));
        let previous = match id {
            "A" => "Start",
            "B" => "A",
            "C" => "A",
            "D" => "B",
            _ => "Start"
        };
        node.add_dep(previous.to_string());
        if id == "D" {
            node.add_dep("C".to_string());
        }
        nodes.insert(id.to_string(), node);
    }
    let mut dep_graph = DepGraph::new(nodes);
    let final_result = match dep_graph.run().unwrap() {
        Output::Tensor32(vec) => vec
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


}

