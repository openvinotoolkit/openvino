// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/variant.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief Constant folding iterates over the function and tries to evaluate nodes
 *        with constant inputs. Such nodes are then replaced with new Constants containing
 *        the result of a folded operation.
 */
class OPENVINO_API ConstantFolding : public FunctionPass {
public:
    OPENVINO_RTTI("ConstantFolding");
    bool run_on_function(std::shared_ptr<ov::Function> f) override;

private:
    void copy_runtime_info_to_target_inputs(const std::shared_ptr<Node>& node, const Output<Node>& replacement);
    /// \brief Folds pre-calculated output tensor values to constants in case lower and
    /// upper estimations are equal. Traverses graph backwards starting from the results.
    bool pre_calculated_values_folding(const std::shared_ptr<ov::Function>& f);
};

OPENVINO_API void disable_constant_folding(const std::shared_ptr<Node>& node);

OPENVINO_API void enable_constant_folding(const std::shared_ptr<Node>& node);

OPENVINO_API bool constant_folding_is_disabled(const std::shared_ptr<Node>& node);

class OPENVINO_API DisableConstantFolding : public VariantImpl<bool> {
public:
    OPENVINO_RTTI("disabled_constant_folding", "0");

    DisableConstantFolding() = default;

    DisableConstantFolding(const value_type& value) : VariantImpl<value_type>(value) {}

    bool is_copyable() const override {
        return false;
    }
};
}  // namespace pass
}  // namespace ov
