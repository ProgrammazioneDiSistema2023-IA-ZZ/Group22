import numpy as np
import onnx_rust2py as onnx

print("Running MNIST-7")
dep_graph = onnx.DepGraph("../src/mnist-7/model.onnx")
input_arr = onnx.parse_input_tensor("../src/mnist-7/test_data_set_0/input_0.pb")
out = dep_graph.py_run(input_arr)
result = onnx.parse_input_tensor("../src/mnist-7/test_data_set_0/output_0.pb")
print(f"Output: {out.as_vec()}")
print(f"Result: {result.as_vec()}")
diff = [a - b for a, b in zip(out.as_vec(), result.as_vec())]
print(f"Difference: {diff}")

print("Running Googlenet")
dep_graph = onnx.DepGraph("../src/googlenet/model.onnx")
input_arr = onnx.parse_input_tensor("../src/googlenet/test_data_set_0/input_0.pb")
out = dep_graph.py_run(input_arr)
result = onnx.parse_input_tensor("../src/googlenet/test_data_set_0/output_0.pb")
print(f"Output: {out.as_vec()}")
print(f"Result: {result.as_vec()}")
diff = [a - b for a, b in zip(out.as_vec(), result.as_vec())]
print(f"Difference: {diff}")

print("Running MNIST-7 with custom input")
dep_graph = onnx.DepGraph("../src/mnist-7/model.onnx")
input_arr = onnx.PyInput.from_numpy(np.full((1, 1, 28, 28), 0.7, dtype=np.float32))
out = dep_graph.py_run(input_arr)
result = onnx.parse_input_tensor("../src/mnist-7/test_data_set_0/output_0.pb")
print(f"Output: {out.as_vec()}")



