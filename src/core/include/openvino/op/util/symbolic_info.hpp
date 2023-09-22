// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/descriptor/tensor.hpp>

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/visibility.hpp"

namespace ov {

OPENVINO_API void set_up_symbolic_info(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<ov::TableOfEquivalence>& table);
OPENVINO_API void set_up_symbolic_info(const ov::Output<ov::Node>& output,
                                       const std::shared_ptr<ov::TableOfEquivalence>& table);

OPENVINO_API void populate_tensor_with_missing_labels(ov::descriptor::Tensor& tensor);

OPENVINO_API bool skip_invalidation(const ov::descriptor::Tensor& tensor);
OPENVINO_API std::shared_ptr<ov::TableOfEquivalence> table_of_equivalence(const std::shared_ptr<ov::Model>& model);
OPENVINO_API std::shared_ptr<ov::TableOfEquivalence> table_of_equivalence(const ov::descriptor::Tensor& tensor);

OPENVINO_API void remove_symbolic_info(const std::shared_ptr<ov::Model>& model, bool outermost_model = true);

/**
 * @ingroup ie_runtime_attr_api
 * @brief SymbolicInfo class represents runtime info attribute that instructs ov::Output objects to skip invalidation of
 * partial values and labels during partial value propagation and keeps shared_ptr to TableOfEquivalence.
 */
class OPENVINO_API SymbolicInfo : public RuntimeAttribute {
public:
    OPENVINO_RTTI("SymbolicInfo", "0");
    explicit SymbolicInfo(bool skip_invalidation, const std::shared_ptr<ov::TableOfEquivalence>& table)
        : m_skip_invalidation{skip_invalidation},
          m_table{table} {};
    bool is_copyable() const override {
        return false;
    }
    bool get_skip_invalidation() const {
        return m_skip_invalidation;
    }
    std::shared_ptr<ov::TableOfEquivalence> get_table() const {
        return m_table;
    }

private:
    bool m_skip_invalidation;
    std::shared_ptr<ov::TableOfEquivalence> m_table;
};

}  // namespace ov
