// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace GNAPluginNS {
/// \brief GNA specific copy layer operation
///
class Copy : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    Copy();
    /// \brief Constructs an Copy operation.
    ///
    /// \param [in] arg Input tensor
    /// \param [in] is_delayed_copy The type for copy layer
    Copy(const ngraph::Output<ngraph::Node>& arg, bool is_delayed_copy);

    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
    bool is_delayed_copy() const {
        return m_is_delayed_copy;
    }

private:
    ///
    /// \brief Control whether simple copy layer or delayed one
    ///
    /// \note The copy layer is executing as is. The delayed copy is
    ///       used in cases, when the result of computation
    ///       is not needed immediately on inference, like in memory cases.
    ///
    bool m_is_delayed_copy;
};
}  // namespace GNAPluginNS
