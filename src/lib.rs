mod onnx_proto3;
mod node;
mod graph;
mod onnx_runtime;
mod binding_python;
mod operations;


#[cfg(test)]
mod tests {
    // This line imports the tests from the tests/general_tests.rs file
    //#[path = "./tests/general_tests.rs"]
    mod general_tests;
}


