use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use crate::onnx_proto3::ModelProto;
use protobuf::{Message};
use crate::conv::Conv;
use ndarray::Array4;

mod conv;
mod onnx_proto3;

pub trait Compute {
    type Item;

    fn compute(&mut self, inputs: Vec<&Self::Item>) -> Self::Item;
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
    let mut inputs = Vec::<&Array4<f32>>::new();
    let first_input = Array4::from_elem((64,3,256,256), 1.3);
    inputs.push(&first_input);
    println!("{}", conv_node.compute(inputs))


}

