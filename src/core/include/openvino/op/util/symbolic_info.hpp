// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/visibility.hpp"

namespace ov {

OPENVINO_API void skip_invalidation(const ov::Output<ov::Node>& output);

OPENVINO_API bool skip_invalidation(const ov::descriptor::Tensor& tensor);

OPENVINO_API void remove_skip_invalidation_rti(const std::shared_ptr<ov::Model>& model, bool outermost_model = true);

OPENVINO_API void populate_tensor_with_missing_symbols(ov::descriptor::Tensor& tensor);

/**
 * @ingroup ov_runtime_attr_api
 * @brief SkipInvalidation class represents runtime info attribute that instructs ov::Output objects to skip
 * invalidation of partial values and symbols during partial value propagation.
 */
class OPENVINO_API SkipInvalidation : public RuntimeAttribute {
public:
    OPENVINO_RTTI("SkipInvalidation", "0", RuntimeAttribute);
    SkipInvalidation() = default;
    ~SkipInvalidation() override;
    bool is_copyable() const override {
        return false;
    }
};

/**
 * @ingroup ov_runtime_attr_api
 * @brief ForceInvalidation class represents runtime info attribute that forces bounds invalidation
 * even when SkipInvalidation is set. Used when input source changes and bounds need recalculation.
 * The attribute is automatically removed after invalidation occurs.
 */
class OPENVINO_API ForceInvalidation : public RuntimeAttribute {
public:
    OPENVINO_RTTI("ForceInvalidation", "0", RuntimeAttribute);
    ForceInvalidation() = default;
    ~ForceInvalidation() override;
    bool is_copyable() const override {
        return false;
    }
};

/// \brief Sets ForceInvalidation attribute on tensor to force bounds invalidation on next invalidate_values() call.
OPENVINO_API void force_invalidation(ov::descriptor::Tensor& tensor);

}  // namespace ov
