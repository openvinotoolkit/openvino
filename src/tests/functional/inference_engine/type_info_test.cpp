// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ngraph_functions/subgraph_builders.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset.hpp"

using TypeInfoTest = ::testing::TestWithParam<std::string>;

inline ov::Core create_core_with_template() {
    ov::Core ie;
#ifndef OPENVINO_STATIC_LIBRARY
    std::string pluginName = "openvino_template_plugin";
    pluginName += IE_BUILD_POSTFIX;
    ie.register_plugin(pluginName, "TEMPLATE");
#endif // !OPENVINO_STATIC_LIBRARY
    return ie;
}

/* The test checks that after plugin unloading
 * creation of opsets doesn't lead to segfault.
 * The rootcause was in use of RTTI 
 * `get_type_info_static()` symbols from 
 * plugins libraries which were linked with 
 * core library instead of directly from core library.
 * Note: issue is reproduced with enabled LTO and
 * `get_type_info_static()` methods with default 
 * visibility instead of hidden.
 */
TEST_P(TypeInfoTest, NoSegFaultOnGetOpSetAfterPluginsUnloading) {
    {
        const auto& device = GetParam();

        ov::Core core = create_core_with_template();
        auto simple_model = ngraph::builder::subgraph::makeSingleConv();
        ov::CompiledModel compiled_net = core.compile_model(simple_model, device);
        auto supported_properties = compiled_net.get_property(ov::supported_properties);
    }
    {
        std::vector<ov::OpSet> opsets;
        opsets.push_back(ov::get_opset1());
        opsets.push_back(ov::get_opset2());
        opsets.push_back(ov::get_opset3());
        opsets.push_back(ov::get_opset4());
        opsets.push_back(ov::get_opset5());
        opsets.push_back(ov::get_opset6());
        opsets.push_back(ov::get_opset7());
        opsets.push_back(ov::get_opset8());
    }
}

INSTANTIATE_TEST_SUITE_P(
        TypeInfoTest, TypeInfoTest,
        ::testing::Values("CPU", "HETERO:CPU", "MULTI:CPU", "AUTO:CPU",
                          "GPU", "HETERO:GPU", "MULTI:GPU", "AUTO:GPU", "BATCH:GPU",
                          "TEMPLATE", "HETERO:TEMPLATE", "MULTI:TEMPLATE", "AUTO:TEMPLATE"));