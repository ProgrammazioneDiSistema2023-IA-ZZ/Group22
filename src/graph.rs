use crossbeam::channel::{unbounded, Sender, Receiver};
use std::collections::{HashMap, HashSet};
use std::{error, mem, thread};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::sync::mpsc::channel;
use ndarray::Array4;
use crate::node::{Node, SimpleNode};
use crate::operations::Output;


pub enum Error {
    NodeNotfound
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Node Not Found")
    }
}

impl Debug for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Node Not Found")
    }
}

impl error::Error for Error {}

pub type InnerDependencyMap = HashMap<String, HashSet<String>>;
pub type DependencyMap = Arc<RwLock<InnerDependencyMap>>;

/// Dependency graph
pub struct DepGraph
{
    pub ready_nodes: Vec<String>,
    pub deps: DependencyMap,
    pub rdeps: DependencyMap,
    pub nodes: Arc<HashMap<String, Arc<RwLock<Node>>>>
}

impl DepGraph
{
    /// Create a new DepGraph based on a vector of edges.
    pub fn new(nodes: HashMap<String, Node>) -> Self {
        let (deps, rdeps, ready_nodes) = DepGraph::parse_nodes(&nodes);
        let nodes_safe = Arc::new(nodes
            .into_iter()
            .map( |(key, val)| (key, Arc::new(RwLock::new(val)))).collect());
        DepGraph {
            ready_nodes,
            deps,
            rdeps,
            nodes: nodes_safe
        }
    }

    fn parse_nodes(nodes_map: &HashMap<String, Node>) -> (DependencyMap, DependencyMap, Vec<String>) {
        let nodes = nodes_map.values()
            .map(|node| SimpleNode::new(node.id().clone(), node.deps().clone()))
            .collect::<Vec<SimpleNode>>();
        let mut deps = InnerDependencyMap::default();
        let mut rdeps = InnerDependencyMap::default();
        let mut ready_nodes = Vec::<String>::default();

        for node in nodes {
            deps.insert(node.id().clone(), node.deps().clone());

            if node.deps().is_empty() {
                ready_nodes.push(node.id().clone());
            }

            for node_dep in node.deps() {
                if !rdeps.contains_key(node_dep) {
                    let mut dep_rdeps = HashSet::new();
                    dep_rdeps.insert(node.id().clone());
                    rdeps.insert(node_dep.clone(), dep_rdeps.clone());
                } else {
                    let dep_rdeps = rdeps.get_mut(node_dep).unwrap();
                    dep_rdeps.insert(node.id().clone());
                }
            }
        }

        (
            Arc::new(RwLock::new(deps)),
            Arc::new(RwLock::new(rdeps)),
            ready_nodes,
        )
    }

    pub fn run(&mut self) -> Option<Output> {
        //Main thread works as dispatcher
        let mut threads = Vec::new();
        let (tx_input, rx_input): (crossbeam::channel::Sender<String>, crossbeam::channel::Receiver<String>) = unbounded();
        let (tx_output, rx_output): (crossbeam::channel::Sender<String>, crossbeam::channel::Receiver<String>)= unbounded();
        for _ in 0..4 {
            let rx = rx_input.clone();
            let tx = tx_output.clone();
            let node_map = self.nodes.clone();
            let thread = thread::spawn(move | | {
                    while let Ok(node) = rx.recv() {
                        let mut node_to_process = node_map.get(&node).unwrap();
                        node_to_process.write().unwrap().compute_operation(&node_map);
                        tx.send(node).unwrap();
                    }
            });
            threads.push(thread);
        }
        //Start Injection
        while let Some(node_to_process) = self.ready_nodes.pop(){
            tx_input.send(node_to_process).unwrap();
        }
        //Running
        let mut last_node = None;
        while let Ok(node_to_add) = rx_output.recv() {
            self.ready_nodes.append(&mut remove_node_id(node_to_add.clone(), &self.deps, &self.rdeps).unwrap());
            //No more nodes leave
            if self.deps.read().unwrap().is_empty() {
                last_node = Some(node_to_add);
                break;
            }
            while let Some(node_to_process) = self.ready_nodes.pop(){
                tx_input.send(node_to_process).unwrap();
            }
        }

        mem::drop(tx_input);
        for handle in threads {handle.join().unwrap();}
        if let Some(inference) = last_node {
            return self.nodes.get(&inference).unwrap().read().unwrap().output.clone();
        }else{
            return None;
        }
    }


}

/// Remove all references to the node ID in the dependencies.
///
pub fn remove_node_id(
    id: String,
    deps: &DependencyMap,
    rdeps: &DependencyMap,
) -> Result<Vec<String>, Error>
{
    let rdep_ids = {
        match rdeps.read().unwrap().get(&id) {
            Some(node) => node.clone(),
            // If no node depends on a node, it will not appear
            // in rdeps.
            None => Default::default(),
        }
    };

    let mut deps = deps.write().unwrap();
    let next_nodes = rdep_ids
        .iter()
        .filter_map(|rdep_id| {
            let rdep = match deps.get_mut(&*rdep_id) {
                Some(rdep) => rdep,
                None => return None,
            };

            rdep.remove(&id);

            if rdep.is_empty() {
                Some(rdep_id.clone())
            } else {
                None
            }
        })
        .collect();

    // Remove the current node from the list of dependencies.
    deps.remove(&id);

    Ok(next_nodes)
}