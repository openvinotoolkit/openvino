// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/descriptor/tensor.hpp>

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/visibility.hpp"

namespace ov {

OPENVINO_API void set_up_symbolic_info(const ov::Output<ov::Node>& output);

OPENVINO_API void populate_tensor_with_missing_symbols(ov::descriptor::Tensor& tensor);

OPENVINO_API bool skip_invalidation(const ov::descriptor::Tensor& tensor);

OPENVINO_API void remove_symbolic_info(const std::shared_ptr<ov::Model>& model, bool outermost_model = true);

/**
 * @ingroup ov_runtime_attr_api
 * @brief SymbolicInfo class represents runtime info attribute that instructs ov::Output objects to skip invalidation of
 * partial values and symbols during partial value propagation.
 */
class OPENVINO_API SymbolicInfo : public RuntimeAttribute {
public:
    OPENVINO_RTTI("SymbolicInfo", "0");
    explicit SymbolicInfo(bool skip_invalidation) : m_skip_invalidation{skip_invalidation} {};
    bool is_copyable() const override {
        return false;
    }
    bool get_skip_invalidation() const {
        return m_skip_invalidation;
    }

private:
    bool m_skip_invalidation;
};

}  // namespace ov
