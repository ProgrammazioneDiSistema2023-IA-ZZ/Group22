mod onnx_proto3;
mod node;
mod add;
mod operations;
mod graph;
mod reshape;
mod soft_max;
mod dropout;
mod gemm;
mod concat;
mod maxpool;
mod start;
mod averagepool;
mod local_response_normalization;
mod relu;
mod matmul;

#[cfg(test)]
mod tests {
    // This line imports the tests from the tests/general_tests.rs file
    //#[path = "./tests/general_tests.rs"]
    mod general_tests;
}

