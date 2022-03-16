// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeinfo>

#include "openvino/core/extension.hpp"
#include "openvino/core/node.hpp"

namespace ov {

/**
 * @brief The base interface for OpenVINO operation extensions
 */
class OPENVINO_API EvaluateExtension : public Extension {
public:
    using Ptr = std::shared_ptr<EvaluateExtension>;
    /**
     * @brief Returns the type info of supported operation
     *
     * @return ov::DiscreteTypeInfo
     */
    virtual const ov::DiscreteTypeInfo& get_type_info() const = 0;
    virtual bool support_evaluate(const std::shared_ptr<const ov::Node>& node,
                                  const std::vector<std::type_info>& input_tensor_types = {},
                                  const std::vector<std::type_info>& output_tensor_types = {}) const = 0;
    virtual bool evaluate(const std::shared_ptr<const ov::Node>& node,
                          ov::TensorVector& output_values,
                          const ov::TensorVector& input_values) const = 0;
    virtual bool evaluate(const std::shared_ptr<const ov::Node>& node,
                          ov::TensorVector& output_values,
                          const ov::TensorVector& input_values,
                          const ov::EvaluationContext& evaluationContext) const {
        return evaluate(node, output_values, input_values);
    }

    /**
     * @brief Destructor
     */
    ~EvaluateExtension() override;

protected:
    bool is_host_tensors(const ov::TensorVector& tensors) const;
};

#define CHECK_TENSOR_TYPES(NODE, INPUTS, OUTPUTS, TENSOR_TYPE) \
    if (!INPUTS.empty()) {                                     \
        if (NODE->get_input_size() != INPUTS.size())           \
            return false;                                      \
        for (const auto& tensor_type : INPUTS) {               \
            if (tensor_type != typeid(TENSOR_TYPE))            \
                return false;                                  \
        }                                                      \
    }                                                          \
    if (!OUTPUTS.empty()) {                                    \
        if (NODE->get_output_size() != OUTPUTS.size())         \
            return false;                                      \
        for (const auto& tensor_type : OUTPUTS) {              \
            if (tensor_type != typeid(TENSOR_TYPE))            \
                return false;                                  \
        }                                                      \
    }

}  // namespace ov
