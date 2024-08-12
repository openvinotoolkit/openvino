// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace weights {

class Bank {
public:
    Bank() = default;

    ov::Tensor get(const ov::Tensor& tensor);

private:
    // Note: suits both - remote and ordinary tensors
    std::unordered_map<void*, ov::Tensor> m_bank;
    std::mutex m_mutex;
};

std::shared_ptr<Bank> bank(const std::string& bank_name);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
