use std::fs::File;
use std::io::Read;
use pyo3::prelude::*;
use ndarray::{ArrayD, IxDyn};
use crate::operations::{Input, Output};
use crate::onnx_runtime::onnxruntime::{Error, parse_from_raw_data};
use std::path::Path;
use numpy::{PyArray, PyReadonlyArrayDyn};
use protobuf::Message;
use crate::onnx_runtime;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use pyo3::types::PyList;
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

    #[staticmethod]

    pub fn from_numpy(py: Python, array: &PyAny) -> PyResult<Self> {
        if let Ok(numpy_array) = array.extract::<PyReadonlyArrayDyn<f32>>() {
            let input = Input::TensorD(numpy_array.to_owned_array());
            return Ok(PyInput { input });
        }

        if let Ok(py_list) = array.extract::<&PyList>() {
            let mut tensor32vec = Vec::new();
            for item in py_list.iter() {
                let arr: &PyArray<f32, _> = item.extract::<&PyArray<f32, _>>()?;
                let rust_array = arr.to_owned_array();
                tensor32vec.push(rust_array);
            }

            return Ok(PyInput { input: Input::Tensor32Vec(tensor32vec) });
        }

        Err(PyErr::new::<PyRuntimeError, _>("Input must be a numpy array or a list of numpy arrays"))
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