// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "internal_properties.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "subgraph_mha.hpp"

namespace ov::test::snippets {

namespace {

// The int8 MHA body with the Softmax quantized on a softmax's theoretical [0, 1] range rather than
// on a calibrated one -- the only in-tree graph the deferred normalization fires on.
const std::vector<ov::PartialShape> mha_int8_unit_range_shapes{{1, 128, 16, 64},
                                                               {1, 128, 16, 64},
                                                               {1, 16, 1, 1},
                                                               {1, 128, 16, 64}};

size_t count_subgraphs(const ov::CompiledModel& compiled_model) {
    size_t subgraphs = 0;
    for (const auto& op : compiled_model.get_runtime_model()->get_ops()) {
        const auto& rt_info = op->get_rt_info();
        const auto layer_type = rt_info.find(ov::exec_model_info::LAYER_TYPE);
        if (layer_type == rt_info.end()) {
            ADD_FAILURE() << "no layerType on " << op->get_friendly_name();
            continue;
        }
        subgraphs += layer_type->second.as<std::string>() == "Subgraph";
    }
    return subgraphs;
}

std::vector<float> infer(const ov::AnyMap& properties, size_t& subgraphs) {
    const auto model = MHAINT8MatMulFunction(mha_int8_unit_range_shapes, 1.F).getOriginal();

    ov::AnyMap config{{"SNIPPETS_MODE", "IGNORE_CALLBACK"}, ov::hint::inference_precision(ov::element::f32)};
    config.insert(properties.begin(), properties.end());
    auto compiled_model =
        ov::test::utils::PluginCache::get().core()->compile_model(model, ov::test::utils::DEVICE_CPU, config);
    subgraphs = count_subgraphs(compiled_model);

    auto request = compiled_model.create_infer_request();
    for (size_t i = 0; i < compiled_model.inputs().size(); ++i) {
        const auto& input = compiled_model.inputs()[i];
        // Seeded per input, so the three rounds see identical data and the four inputs do not.
        request.set_tensor(input,
                           ov::test::utils::create_and_fill_tensor(
                               input.get_element_type(),
                               input.get_shape(),
                               ov::test::utils::InputGenerateData(-1, 2, 256, static_cast<int32_t>(i) + 1)));
    }
    request.infer();

    const auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    const auto* data = output.data<float>();
    return {data, data + output.get_size()};
}

size_t count_differing(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    EXPECT_EQ(lhs.size(), rhs.size());
    size_t differing = 0;
    for (size_t i = 0; i < lhs.size() && i < rhs.size(); ++i) {
        differing += lhs[i] != rhs[i];
    }
    return differing;
}

}  // namespace

// Nothing else in the suite crosses the boundary between a plugin property and this option: the
// pass tests construct SoftmaxDecomposition directly, and every MHA functional test runs with the
// default. The hop has to be checked by running the model rather than by reading the flag back,
// because a dropped opt-in still compiles and still computes a softmax -- reading it back would
// pass just as happily against a pipeline that never consulted it.
TEST(smoke_Snippets_DeferSoftmaxNormalization, PropertyReachesTheSnippetsPass) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    size_t default_subgraphs = 0;
    size_t off_subgraphs = 0;
    size_t on_subgraphs = 0;
    const auto by_default = infer({}, default_subgraphs);
    const auto normalized = infer({{ov::intel_cpu::snippets_defer_softmax_normalization.name(), false}}, off_subgraphs);
    const auto deferred = infer({{ov::intel_cpu::snippets_defer_softmax_normalization.name(), true}}, on_subgraphs);

    // Without a tokenized body there is no pass to reach, and the comparisons below would hold or
    // fail for an unrelated reason. How many bodies this graph tokenizes into is not this test's
    // business -- smoke_Snippets_MHAINT8MatMulUnitRange already pins that -- only that it does.
    ASSERT_GT(default_subgraphs, 0U);
    ASSERT_GT(off_subgraphs, 0U);
    ASSERT_GT(on_subgraphs, 0U);

    // Not setting the property and setting it to false must be the same thing, which is what pins
    // the default now that a plugin can move it.
    EXPECT_EQ(count_differing(by_default, normalized), 0U);

    // Deferring re-quantizes MatMul1's left operand, so on this body the answer moves over most of
    // the output: 112896 of 131072 elements as measured on avx2+avx_vnni. A floor rather than a
    // bare inequality, so that a change leaving only a handful of elements moving fails here
    // instead of squeaking through. The bar is set well under the measured share because the exact
    // count is a quantization coincidence rate and may shift with the brgemm accumulation order on
    // another ISA; removing the opt-in entirely takes it to 0, which is what it has to catch.
    EXPECT_GT(count_differing(normalized, deferred), deferred.size() / 8);
}

}  // namespace ov::test::snippets
