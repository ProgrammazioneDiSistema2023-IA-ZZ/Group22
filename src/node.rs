use std::cmp::PartialEq;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use ndarray::Array4;
use crate::operations::{Compute, Input, Output};

/// Single node in a dependency graph, which might have dependencies or be
/// be used as a dependency by other nodes.
///
/// A node is represented by a unique identifier and may contain a list of
/// dependencies.
pub struct Node
{
    id: String,
    deps: Vec<String>,
    operation: Box<dyn Compute + Send + Sync>,
    pub output: Option<Output>
}

///utility struct to generate dependency Graph
pub struct SimpleNode {
    id: String,
    deps: HashSet<String>,
}

impl SimpleNode
{
    pub fn new(id: String, deps: HashSet<String>) -> SimpleNode {
        SimpleNode {
            id,
            deps,
        }
    }
    pub fn id(&self) -> String {
        self.id.clone()
    }
    pub fn deps(&self) -> &HashSet<String> {
        &self.deps
    }
}

impl Node
{
    pub fn new(id: String, operation: Box<dyn Compute + Send + Sync>) -> Node {
        Node {
            id,
            deps: Vec::default(),
            operation,
            output: None
        }
    }

    pub fn id(&self) -> String {
        self.id.clone()
    }
    pub fn deps(&self) -> HashSet<String> {

        HashSet::from_iter(self.deps.clone().into_iter())
    }
    pub fn add_dep(&mut self, dep: String) {
        self.deps.push(dep);
    }

    pub fn compute_operation(&mut self, nodes: &HashMap<String, Arc<RwLock<Node>>>) -> () {
        //TODO Remember to handle gemm node order of input
        if self.deps.len() == 1{
            let elem = self.deps.iter().next().unwrap().clone();
            let only_dep = nodes.get(&elem).unwrap();
            let input = match only_dep.read().unwrap().output.clone().unwrap() {
                Output::Tensor32(array) => Input::Tensor32(array),
                _ => panic!("wrong output")
            };
            self.output = Some(self.operation.compute(input));

        }else if self.deps.len() > 1 {
            let mut inputs = Vec::<Array4<f32>>::new();
            self.deps.iter().for_each(|dep| {
                let elem = nodes.get(dep).unwrap();
                let input = match elem.read().unwrap().output.clone().unwrap() {
                    Output::Tensor32(array) => array,
                    _ => panic!("wrong output")
                };
                inputs.push(input);
            });
            self.output = Some(self.operation.compute(Input::Tensor32Vec(inputs)));

        }else{
            self.output = Some(self.operation.compute(Input::Empty));
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let id = self.id();
        let inputs = self.deps.iter()
            .map(|v| (*v).clone()).reduce(|d1, d2| d1 + " --- " + d2.as_str()).unwrap();
        let op_type = self.operation.op_type();
        write!(f, "id: {}\nop_type: {}\ninputs: {}\n", id, op_type, inputs)
    }
}
