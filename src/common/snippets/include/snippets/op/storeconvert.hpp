// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include "store.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface StoreConvert
 * @brief Generated for store and convert at the same time
 * @ingroup snippets
 */
class StoreConvert : public Store {
public:
    OPENVINO_OP("StoreConvert", "SnippetsOpset", ngraph::snippets::op::Store);

    StoreConvert(const Output<Node>& x, const ov::element::Type& destination_type, const size_t lanes = 0lu);
    StoreConvert() = default;

    ov::element::Type get_destination_type() const { return m_destination_type; }

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool has_evaluate() const override { return false; }

protected:
    ov::element::Type m_destination_type;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
