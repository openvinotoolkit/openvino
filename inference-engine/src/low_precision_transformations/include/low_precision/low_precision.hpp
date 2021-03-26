// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/layer_transformation.hpp"
#include "low_precision/markup_precisions.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API LowPrecision;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

//template <typename T>
//std::pair<std::string, std::vector<uint64_t>> getKey(const size_t port, const bool specifiedVersion = false) {
//    const ngraph::Node::type_info_t& typeInfo = T::get_type_info_static();
//    return { typeInfo.name, specifiedVersion ? std::vector<uint64_t>{typeInfo.version} : std::vector<uint64_t>{} };
//}

class ngraph::pass::low_precision::LowPrecision: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LowPrecision(
        const std::vector<OperationPrecisionRestriction>& restrictions = {},
        const LayerTransformation::Params = LayerTransformation::Params());
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

protected:
    std::vector<OperationPrecisionRestriction> restrictions;
    // remove
    LayerTransformation::Params params;
};
