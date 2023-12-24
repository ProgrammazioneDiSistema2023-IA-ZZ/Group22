use std::collections::HashMap;
use crate::graph::DepGraph;
use crate::node::Node;

//Interface struct for
pub mod onnxruntime {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Read;
    use ndarray::{ArrayD, IxDyn};
    use protobuf::Message;
    use crate::averagepool::AveragePool;
    use crate::concat::Concat;
    use crate::Conv::Conv;
    use crate::dropout::Dropout;
    use crate::gemm::Gemm;
    use crate::local_response_normalization::LRN;
    use crate::maxpool::MaxPool;
    use crate::node::Node;
    use crate::onnx_proto3::{GraphProto, ModelProto};
    use crate::operations::{Compute, Output};
    use crate::relu::Relu;
    use crate::reshape::Reshape;
    use crate::soft_max::SoftMax;
    use crate::start::Start;

    #[derive(Debug)]
    pub enum Error{
        ProtoBufError
    }

    pub fn parse_onnx(path: String) -> Result<ModelProto, Error>{
        let mut input_onnx = File::open(path.as_str()).unwrap();
        //Onnx file into byte array
        let mut byte_array = Vec::<u8>::new();
        input_onnx.read_to_end(&mut byte_array).unwrap();
        //Parsing del byte array nella struttura onnx_proto3.rs
        let model: ModelProto = match Message::parse_from_bytes(&byte_array) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("Failed to parse the ONNX model: {}", err);
                return Err(Error::ProtoBufError);
            }
        };
        return Ok(model);
    }
    pub fn get_computational_graph(path: String) -> HashMap<String, Node>{
        let model = parse_onnx(path).unwrap();
        let graph = model.get_graph();
        let nodes = get_nodes(graph);
        return nodes.into_iter().map(|n| (n.id(), n)).collect::<HashMap<String, Node>>();
    }

    pub fn parse_initializers(graph: &GraphProto) -> Vec<Node>{
        let starting_nodes = graph.get_initializer();
        let mut cnt = 0;
        let nodes: Vec<Node> = starting_nodes.into_iter().map(|tensor| {
            cnt += 1;
            let dims: Vec<usize> = tensor.get_dims().iter().map(|val| *val as usize).collect();
            let mut raw = tensor.get_raw_data();
            let mut data: Vec<f32> = Vec::new();
            match tensor.get_data_type() {
                1 => {
                    if raw.len() != 0 {
                        data = parse_from_raw_data(raw);
                    } else {
                        data = tensor.get_float_data().into_iter().map(|val| *val).collect();
                    }
                },
                7 => data = tensor.get_int64_data().into_iter().map(|val| *val as f32).collect(),
                _ => ()
            }
            let tensor_d = ArrayD::from_shape_vec(IxDyn(&dims), data).unwrap();
            let mut tmp_node = Node::new(tensor.name.clone(), Box::new(Start::new()));
            tmp_node.output = Some(Output::TensorD(tensor_d));
            return tmp_node
        }).collect();
        println!("All parsed = {}",starting_nodes.len() == nodes.len());
        return nodes;
    }

    pub fn parse_from_raw_data(raw: &[u8]) -> Vec<f32>{
        return raw.chunks_exact(4) // Split into chunks of 4 bytes (size of f32)
            .map(|chunk| {
                let mut bytes_array = [0; 4];
                bytes_array.copy_from_slice(chunk);
                f32::from_bits(u32::from_le_bytes(bytes_array)) // Convert u8 to f32
            })
            .collect();
    }

    pub fn get_in_out_mapping(graph: &GraphProto) -> HashMap<String, String>{
        let nodes = graph.get_node();
        return nodes.into_iter().flat_map(|x| {
            let name = x.name.clone();
            return x.get_output().into_iter().map(|s| (s.clone(), name.clone())).collect::<Vec<(String, String)>>();
        }).collect::<HashMap<String, String>>();
    }

    pub fn get_nodes(graph: &GraphProto) -> Vec<Node>{
        let alias = get_in_out_mapping(graph);
        return graph.get_node().into_iter().map(|node| {
            let id = node.name.clone();
            let res: Box<dyn Compute + Send + Sync> = match node.get_op_type(){
                    "Softmax" => Box::new(SoftMax::parse_from_proto_node(node.get_attribute())),
                    "Relu" => Box::new(Relu::parse_from_proto_node(node.get_attribute())),
                    "Concat" => Box::new(Concat::parse_from_proto_node(node.get_attribute())),
                    "Dropout" => Box::new(Dropout::parse_from_proto_node(node.get_attribute())),
                    "MaxPool" => Box::new(MaxPool::parse_from_proto_node(node.get_attribute())),
                    "LRN" => Box::new(LRN::parse_from_proto_node(node.get_attribute())),
                    "AveragePool" => Box::new(AveragePool::parse_from_proto_node(node.get_attribute())),
                    "Conv" => Box::new(Conv::parse_from_proto_node(node.get_attribute())),
                    "Reshape" => Box::new(Reshape::parse_from_proto_node(node.get_attribute())),
                    "Gemm" => Box::new(Gemm::parse_from_proto_node(node.get_attribute())),
                    _ => panic!("Unknown operation type!")
                };
            let mut new_node = Node::new(id, res);
            for dep in node.get_input(){
                let mut tmp = dep;
                if alias.contains_key(dep) {
                    tmp = alias.get(dep).unwrap();
                }
                new_node.add_dep((*tmp).clone());
            }
            return new_node;
        }).collect::<Vec<Node>>();
    }

}