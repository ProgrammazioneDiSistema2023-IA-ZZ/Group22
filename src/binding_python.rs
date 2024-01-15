use std::fs::File;
use std::io::Read;
use pyo3::prelude::*;
use ndarray::{Array1, Array4, ArrayD, IxDyn};
use crate::operations::{Input, Output};
use crate::onnx_runtime::onnxruntime::{Error, parse_from_raw_data};
use std::path::Path;
use protobuf::Message;
use crate::onnx_runtime;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use crate::onnx_proto3::TensorProto;
use crate::onnx_runtime::onnxruntime::Error as OnnxRuntimeError;

impl From<OnnxRuntimeError> for PyErr {
    fn from(err: OnnxRuntimeError) -> Self {
        match err {
            OnnxRuntimeError::ProtoBufError => {
                PyErr::new::<PyRuntimeError, _>("ONNX Runtime error: Failed to parse the model (protobuf error)")
            },
            OnnxRuntimeError::InputParsingError => {
                PyErr::new::<PyRuntimeError, _>("ONNX Runtime error: Failed to parse input tensor")
            },
            OnnxRuntimeError::ShapeError => {
                PyErr::new::<PyRuntimeError, _>("ONNX Runtime error: Invalid tensor shape")
            },
            OnnxRuntimeError::ConversionError => {
                PyErr::new::<PyRuntimeError, _>("ONNX Runtime error: Data conversion error")
            },
        }
    }
}

#[pyclass]
struct DepGraph {
    pub dep_graph: crate::graph::DepGraph,
}

#[pymethods]
impl DepGraph {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let model_path = Path::new(model_path);
        let model_path_str = model_path.to_string_lossy().to_string();
        let dep_graph = onnx_runtime::onnxruntime::get_computational_graph(model_path_str);
        Ok(DepGraph { dep_graph })
    }

    fn py_run(&mut self, py_input: PyInput) -> PyResult<Option<PyOutput>> {
        match self.dep_graph.run(py_input.input) {
            Some(output) => {
                // Converti Output in PyOutput (assumendo che tu abbia una conversione adeguata)
                let py_output = PyOutput{ output };
                    Ok(Some(py_output))
            },
            None => Ok(None),
        }
    }

}

#[pyclass]
#[derive(Clone)]
pub struct PyInput {
    pub input: Input,
}

#[pymethods]
impl PyInput {
    pub fn as_vec(&self) -> PyResult<Vec<f32>> {
        match &self.input {
            Input::Tensor32(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::Tensor1(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::Tensor2(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::Tensor3(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::Tensor4(tensor) => Ok(tensor.clone().into_raw_vec()),
            Input::TensorD(tensor) => Ok(tensor.clone().into_raw_vec()),
            _ => Err(PyErr::new::<PyValueError, _>("Unsupported input type")),
        }
    }
    pub fn as_list(&self) -> PyResult<Vec<Vec<f32>>> {
        match &self.input {
            Input::Tensor4List(list) => {
                Ok(Input::Tensor4List(list.clone()).list_into_raw_vec()?)
            },
            Input::Tensor32Vec(list) => {
                Ok(Input::Tensor32Vec(list.clone()).list_into_raw_vec()?)
            },
            _ => Err(PyErr::new::<PyValueError, _>("Unsupported input type")),
        }
    }
}

#[pyfunction]
fn parse_input_tensor(path: String) -> PyResult<PyInput> {
    let mut input_tensor = File::open(&path)
        .map_err(|e| PyIOError::new_err(format!("Failed to open file: {}", e)))?;

    let mut byte_array = Vec::new();
    input_tensor.read_to_end(&mut byte_array)
        .map_err(|e| PyIOError::new_err(format!("Failed to read file: {}", e)))?;

    let input_parsed: TensorProto = match Message::parse_from_bytes(&byte_array) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("Failed to parse the tensor: {}", err);
            return Err(PyIOError::new_err("Failed to parse the tensor"));
        }
    };
    let dims = input_parsed.get_dims()
        .into_iter().map(|v| *v as usize).collect::<Vec<usize>>();
    let vec = parse_from_raw_data(input_parsed.get_raw_data());
    let input = Input::TensorD(ArrayD::from_shape_vec(IxDyn(&dims), vec).unwrap());
    Ok(PyInput { input })
}


#[pyclass]
#[derive(Clone)]
pub struct PyOutput {
    pub output: Output,
}

#[pymethods]
impl PyOutput {
    pub fn as_vec(&self) -> PyResult<Vec<f32>> {
        match &self.output {
            Output::Tensor32(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::Tensor1(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::Tensor2(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::Tensor3(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::Tensor4(tensor) => Ok(tensor.clone().into_raw_vec()),
            Output::TensorD(tensor) => Ok(tensor.clone().into_raw_vec()),
            _ => Err(PyErr::new::<PyValueError, _>("Unsupported input type")),
        }
    }
}

#[pyfunction]
fn diff_out(v1: PyOutput, v2: PyOutput) -> PyResult<Vec<f32>> {
    let v1_raw = v1.output.into_raw_vec()?;
    let v2_raw = v2.output.into_raw_vec()?;
    if v1_raw.len() != v2_raw.len() {
        return Err(PyErr::new::<PyRuntimeError, _>("Error: Vectors have different lengths"));
    }
    let diff = v1_raw.iter().zip(v2_raw.iter()).map(|(a, b)| a - b).collect();
    Ok(diff)
}



#[pymodule]
fn onnx_rust2py(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DepGraph>()?;
    m.add_class::<PyInput>()?;
    m.add_class::<PyOutput>()?;
    m.add_function(wrap_pyfunction!(parse_input_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(diff_out, m)?)?;
    Ok(())
}