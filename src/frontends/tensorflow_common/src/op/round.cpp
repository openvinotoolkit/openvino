// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"
#include<complex>
#include<vector>
#include"boolvariant.hpp"
#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

class ComplexTensor {
public:
    // Constructor
    ComplexTensor(std::vector<std::complex<float>> tensor, std::vector<int> shape)
        : tensor_(tensor), shape_(shape) {}

    // Method to extract the real part of the tensor
    std::vector<float> real() const {
        std::vector<float> real_tensor(tensor_.size());
        for (int i = 0; i < tensor_.size(); ++i) {
            real_tensor[i] = tensor_[i].real();
        }
        return real_tensor;
    }

    // Method to extract the imaginary part of the tensor
    std::vector<float> imag() const {
        std::vector<float> imag_tensor(tensor_.size());
        for (int i = 0; i < tensor_.size(); ++i) {
            imag_tensor[i] = tensor_[i].imag();
        }
        return imag_tensor;
    }

    // Method to get the shape of the tensor
    std::vector<int> shape() const {
        return shape_;
    }

 private:
    std::vector<std::complex<float>> tensor_;
    std::vector<int> shape_;
};

// Flat the tensor
std::vector<std::complex<float>> flatten(const std::vector<std::vector<std::complex<float>>>& tensor) {
    std::vector<std::complex<float>> flat_tensor;
    for (const auto& row : tensor) {
        flat_tensor.insert(flat_tensor.end(), row.begin(), row.end());
    }
    return flat_tensor;
}

std::vector<int> shape(const std::vector<std::vector<std::complex<float>>>& tensor) {
    return {static_cast<int>(tensor.size()), static_cast<int>(tensor[0].size())};
}

OutputVector translate_round_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Round", "ROUND"});

    auto input = node.get_input(0);
    if (input.get_rt_info().count("ComplexTypeMark")) {
        // Propagate the ComplexTypeMark to the output tensor
        auto res = make_shared<v5::Round>(input, v5::Round::RoundMode::HALF_TO_EVEN);
        res->get_rt_info()["ComplexTypeMark"] = std::make_shared<ov::BoolVariant>(true);
    // using default round mode "half_to_even" in openvino,
    // as TF has only that mode
    auto round_mode = v5::Round::RoundMode::HALF_TO_EVEN;
    auto res = make_shared<v5::Round>(input, round_mode);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov