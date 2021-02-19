// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset3.hpp>

#include <memory>
#include <vector>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeTopK : public ngraph::op::v3::TopK {
public:
    NGRAPH_RTTI_DECLARATION;

    StaticShapeTopK(const Output<Node>& data,
                    const Output<Node>& k,
                    const int64_t axis,
                    const std::string& mode,
                    const std::string& sort,
                    const element::Type& index_element_type = element::i32);

    StaticShapeTopK(const Output<Node>& data,
                    const Output<Node>& k,
                    const int64_t axis,
                    const Mode mode,
                    const SortType sort,
                    const element::Type& index_element_type = element::i32);

    void validate_and_infer_types() override;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
