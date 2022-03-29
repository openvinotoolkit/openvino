// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <transformations_visibility.hpp>

//#include "ngraph/coordinate_diff.hpp"
#include "ngraph/opsets/opset9.hpp"

namespace ngraph {
namespace op {
namespace internal {

class TRANSFORMATIONS_API MulticlassNmsIEInternal : public opset9::MulticlassNms {
public:
    NGRAPH_RTTI_DECLARATION;

    MulticlassNmsIEInternal() = default;

    MulticlassNmsIEInternal(const Output<Node>& boxes,
                            const Output<Node>& scores,
                            const ov::op::util::MulticlassNmsBase::Attributes& attrs);

    MulticlassNmsIEInternal(const Output<Node>& boxes,
                            const Output<Node>& scores,
                            const Output<Node>& roisnum,
                            const ov::op::util::MulticlassNmsBase::Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    // const ::ov::DiscreteTypeInfo& get_type_info() const override;

private:
    typedef struct {
    } init_rt_result;

    init_rt_result init_rt_info() {
        opset9::MulticlassNms::get_rt_info()["opset"] = "ie_internal_opset";
        return {};
    }

    init_rt_result init_rt = init_rt_info();
};
}  // namespace internal
}  // namespace op
}  // namespace ngraph
