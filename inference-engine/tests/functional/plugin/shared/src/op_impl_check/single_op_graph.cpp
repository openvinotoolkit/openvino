// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_impl_check/op_impl_check.hpp>

namespace ov {
namespace test {
namespace subgraph {


//using OpGenerator = std::map<ngraph::NodeTypeInfo, std::function<std::shared_ptr<ov::Function>(const ov::DiscreteTypeInfo& typeInfo)>>;

//OpGenerator getOpGeneratorMap();

template<typename T>
InferenceEngine::Blob::Ptr generateInput(const std::shared_ptr<ngraph::Node> node,
                                         const InferenceEngine::InputInfo& info,
                                         size_t port) {
    return generate(ngraph::as_type_ptr<T>(node), info, port);
}
} // namespace

OpGenerator getOpGeneratorMap() {
    static OpGenerator inputsMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), generateInput<NAMESPACE::NAME>},
#include "ngraph/opsets/opset1_tbl.hpp"
#include "ngraph/opsets/opset2_tbl.hpp"
#include "ngraph/opsets/opset3_tbl.hpp"
#include "ngraph/opsets/opset4_tbl.hpp"
#include "ngraph/opsets/opset5_tbl.hpp"
#include "ngraph/opsets/opset6_tbl.hpp"
#include "ngraph/opsets/opset7_tbl.hpp"
#include "ngraph/opsets/opset8_tbl.hpp"
#undef NGRAPH_OP
    };
    return inputsMap;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov