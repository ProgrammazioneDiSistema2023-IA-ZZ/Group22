
use std::collections::{HashMap, HashSet};
use std::error;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use crate::node::{Node, SimpleNode};


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
}

impl DepGraph
{
    /// Create a new DepGraph based on a vector of edges.
    pub fn new(nodes: &HashMap<String, Node>) -> Self {
        let (deps, rdeps, ready_nodes) = DepGraph::parse_nodes(nodes);

        DepGraph {
            ready_nodes,
            deps,
            rdeps,
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