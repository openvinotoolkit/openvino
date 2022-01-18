// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/pass_debug.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ops/gna_convolution.hpp>

#include <vector>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(PassDebug, "PassDebug", 0);

namespace {
#define EMUTEX_DEBUG_VAL(x) std::cout << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " " << #x << " " << x << std::endl;
void print_convolution_constant(std::shared_ptr<ngraph::Function> func)
{
    for (auto node : func->get_ordered_ops())
    {
        auto convolution_node = std::dynamic_pointer_cast<GNAPluginNS::Op::GNAConvolution>(node);
        if (!convolution_node)
            continue;
        auto convolution_input_const_node = convolution_node->input_value(1);
        auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(convolution_input_const_node.get_node_shared_ptr());
        if (!const_node)
            continue;
        const std::vector<float> v = const_node->cast_vector<float>();
        for (size_t i = 0; i < std::min(static_cast<size_t>(20), v.size());  ++i)
        {
            float f = v[i];
            EMUTEX_DEBUG_VAL(f);
        }
        break;
    }
}
} // namespace

bool PassDebug::run_on_function(std::shared_ptr<ngraph::Function> f) {

    print_convolution_constant(f);

    return false;
}

