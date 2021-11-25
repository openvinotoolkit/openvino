// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_impl_check/op_impl_check.hpp>
#include <op_impl_check/single_op_graph.hpp>

namespace ov {
namespace test {
namespace subgraph {


//using OpGenerator = std::map<ngraph::NodeTypeInfo, std::function<std::shared_ptr<ov::Function>(const ov::DiscreteTypeInfo& typeInfo)>>;

//OpGenerator getOpGeneratorMap();

std::shared_ptr<ov::Function> generate(const &ngraph::opset1::Add::get_type_info_static() node) {
    return nullptr;
}

std::function<std::shared_ptr<ov::Node>(const std::vector<ov::op::v0::Parameter>& params,
                                        const ov::DiscreteTypeInfo& typeInfo)>>;

template<typename T>
std::shared_ptr<ov::Node> generateInput(const std::vector<ov::op::v0::Parameter>& params, const ov::DiscreteTypeInfo& typeInfo) {
    return generate(ngraph::as_type_ptr<T>(node), info, port);
}

OpGenerator getOpGeneratorMap() {
    static OpGenerator a{
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
    return a;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov