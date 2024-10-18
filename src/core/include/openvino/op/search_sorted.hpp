// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief SearchSorted operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SearchSorted : public Op {
public:
    OPENVINO_OP("SearchSorted", "opset15", Op);

    SearchSorted() = default;
    /// \brief Constructs a SearchSorted operation.
    /// \param sorted_sequence Sorted sequence to search in.
    /// \param values          Values to search indexs for.
    /// \param right_mode      If False, return the first suitable index that is found for given value. If True, return
    /// the last such index.
    SearchSorted(const Output<Node>& sorted_sequence, const Output<Node>& values, bool right_mode = false);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_right_mode() const {
        return m_right_mode;
    }

    void set_right_mode(bool right_mode) {
        m_right_mode = right_mode;
    }

private:
    bool m_right_mode{};
};
}  // namespace v15
}  // namespace op
}  // namespace ov
