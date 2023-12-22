
/*#[cfg(test)]
pub mod tests {*/
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
use crate::onnx_runtime;
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

    #[test]
    fn test_parse_initializer(){
        let model = onnx_runtime::onnxruntime::parse_onnx("src/gender_googlenet.onnx".to_string()).unwrap();
        let graph = model.get_graph();
        let nodes = onnx_runtime::onnxruntime::parse_initializers(graph);
        let test_len = nodes.len();
        nodes.into_iter().for_each(|node| {
            print!("{} - dim: ", node.id());
            let name = node.id().to_string();
            if let Output::TensorD(array) = node.output.unwrap(){
                array.shape().iter().for_each(|val| print!("{} ", *val));
                if name == "loss3/classifier_agexgender_shape".to_string(){
                    println!("Got it");
                    array.iter().for_each(|val| print!("{} ", *val));
                }
            }
            println!();
        });
        assert_eq!(graph.get_initializer().len(), test_len);
    }
//}