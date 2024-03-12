#pragma once

#include <vector>
#include <complex>
#include "openvino/core/node.hpp"
#include <tensor.hpp>
#include <host_tensor.hpp>

using namespace ov::op;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

std::vector<std::complex<float>> flatten(const ov::Output<ov::Node>& tensor) {
    auto host_tensor_ptr = tensor.get_tensor_ptr();
    auto data_ptr = host_tensor_ptr->get_data_ptr<std::complex<float>>();
    
    // Assuming the tensor is 1D or 2D, calculate the total number of elements.
    auto shape = host_tensor_ptr->get_shape();
    size_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    
    // Create a vector from the data pointer.
    std::vector<std::complex<float>> flat_tensor(data_ptr, data_ptr + num_elements);
    
    return flat_tensor;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov