#pragma once

#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"
#include<tensor.hpp>
#include<concat.hpp>
#include <host_tensor.hpp>
#include<complex>



using namespace ov::op;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector reshape_complex_tensor(const ov::Output<ov::Node>& real, const ov::Output<ov::Node>& imag, const ov::Output<ov::Node>& shape) {
    // Concatenate the shape with [2] to account for the real and imaginary parts.
    OutputVector concat_inputs;
    concat_inputs.push_back(shape);
    concat_inputs.push_back(make_shared<v0::Constant>(shape.get_element_type(), Shape{1}, 2));
    auto concat = make_shared<v0::Concat>(concat_inputs, 0);

    // Reshape the real and imaginary parts.
    auto real_reshape = make_shared<v1::Reshape>(real, concat, false);
    auto imag_reshape = make_shared<v1::Reshape>(imag, concat, false);

    // Wrap the reshaped real and imaginary parts in a ComplexTensor.
    ComplexTensor complexTensor(real_reshape, imag_reshape);

    // Wrap the ComplexTensor in an ov::Node and return it.
    auto complex_node = make_shared<ov::Node>(complexTensor);
    return {complex_node};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov