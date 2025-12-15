// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API OneHotBase : public Op {
public:
    OPENVINO_OP("OneHot", "util");

    /// \brief Constructs a one-hot operation.
    OneHotBase() = default;

    /// \brief Constructs a one-hot operation.
    ///
    /// \param indices   Input tensor containing indices.
    /// \param depth     Specifies number of classes and the size of one-hot dimension.
    /// \param on_value  Specifies value that the locations in output tensor represented
    ///                  by indices in input take.
    /// \param off_value Specifies value that the locations in output tensor not
    /// represented
    ///                  by indices in input take.
    /// \param axis      Axis along which one-hot representation in added.
    OneHotBase(const Output<Node>& indices,
               const Output<Node>& depth,
               const Output<Node>& on_value,
               const Output<Node>& off_value,
               int64_t axis);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return The index of the one-hot axis.
    const int64_t& get_axis() const {
        return m_axis;
    }

    /// @brief Sets the index of the one-hot axis.
    /// @param axis The index of the one-hot axis.
    void set_axis(int64_t axis);

protected:
    int64_t m_axis;

private:
    friend void inline resolve_axis(OneHotBase* op);
};
}  // namespace util
}  // namespace op
}  // namespace ov
