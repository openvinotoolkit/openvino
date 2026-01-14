// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/stateful_sdpa_bool_mask.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/include/common_test_utils/data_utils.hpp"
#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/runtime/system_conf.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string StatefulSdpaBoolMaskTest::getTestCaseName(const testing::TestParamInfo<StatefulSdpaBoolMaskParams>& obj) {
    std::ostringstream result;
    result << "InferenceType=" << obj.param.get_type_name();
    return result.str();
}

void StatefulSdpaBoolMaskTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    const auto inferencePrecision = GetParam();
    const auto graphPrecision = ov::element::f32;

    configuration[ov::hint::inference_precision.name()] = inferencePrecision;
    configuration[ov::hint::kv_cache_precision.name()] = ov::element::f32;
    rel_threshold = 0.02f;
    abs_threshold = 0.02f;
    selectedType = makeSelectedTypeStr(getPrimitiveType(), inferencePrecision);

    const InputShape q_shape{{-1, 8, -1, 64}, {{1, 8, 16, 64}}};
    const InputShape k_shape{{-1, 8, -1, 64}, {{1, 8, 16, 64}}};
    const InputShape v_shape{{-1, 8, -1, 64}, {{1, 8, 16, 64}}};
    const InputShape mask_shape{{1, 1, -1, -1}, {{1, 1, 16, 16}}};
    const InputShape past_shape{{-1, 8, -1, 64}, {{1, 8, 0, 64}}};
    const InputShape beam_shape{{-1}, {{1}}};

    init_input_shapes({q_shape, k_shape, v_shape, mask_shape, past_shape, beam_shape});

    auto q = std::make_shared<ov::op::v0::Parameter>(graphPrecision, inputDynamicShapes[0]);
    auto k = std::make_shared<ov::op::v0::Parameter>(graphPrecision, inputDynamicShapes[1]);
    auto v = std::make_shared<ov::op::v0::Parameter>(graphPrecision, inputDynamicShapes[2]);
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, inputDynamicShapes[3]);
    auto past_init = std::make_shared<ov::op::v0::Parameter>(graphPrecision, inputDynamicShapes[4]);
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[5]);

    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");
    mask->set_friendly_name("attention_mask");
    past_init->set_friendly_name("past_init");
    beam_idx->set_friendly_name("beam_idx");

    auto variable_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputDynamicShapes[4], graphPrecision, "pastk"});
    auto variable_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{inputDynamicShapes[4], graphPrecision, "pastv"});

    auto past_k = std::make_shared<ov::op::v6::ReadValue>(past_init, variable_k);
    auto past_v = std::make_shared<ov::op::v6::ReadValue>(past_init, variable_v);
    past_k->set_friendly_name("pastk_read");
    past_v->set_friendly_name("pastv_read");

    auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
    auto gather_k = std::make_shared<ov::op::v8::Gather>(past_k, beam_idx, axis);
    auto gather_v = std::make_shared<ov::op::v8::Gather>(past_v, beam_idx, axis);
    gather_k->set_batch_dims(0);
    gather_v->set_batch_dims(0);

    auto concat_k = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather_k, k}, 2);
    auto concat_v = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather_v, v}, 2);

    auto sdpa = std::make_shared<ov::opset13::ScaledDotProductAttention>(q, concat_k, concat_v, mask, false);
    sdpa->set_friendly_name("stateful_sdpa");

    auto assign_k = std::make_shared<ov::op::v6::Assign>(concat_k, variable_k);
    auto assign_v = std::make_shared<ov::op::v6::Assign>(concat_v, variable_v);
    assign_k->set_friendly_name("pastk_write");
    assign_v->set_friendly_name("pastv_write");

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(sdpa)};
    ov::SinkVector sinks{assign_k, assign_v};
    function = std::make_shared<ov::Model>(results,
                                           sinks,
                                           ov::ParameterVector{q, k, v, mask, past_init, beam_idx},
                                           "StatefulSdpaBoolMask");
}

void StatefulSdpaBoolMaskTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& parameters = function->get_parameters();
    for (size_t idx = 0; idx < parameters.size(); ++idx) {
        const auto& param = parameters[idx];
        const auto& shape = targetInputStaticShapes[idx];
        if (param->get_element_type() == ov::element::f32) {
            ov::Tensor tensor{ov::element::f32, shape};
            utils::fill_data_random(static_cast<float*>(tensor.data()), tensor.get_size(), 2, -1, 16);
            inputs.insert({param, tensor});
        } else if (param->get_element_type() == ov::element::boolean) {
            ov::Tensor tensor{ov::element::boolean, shape};
            auto* data = tensor.data<bool>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = (i % 3) != 0;
            }
            inputs.insert({param, tensor});
        } else if (param->get_element_type() == ov::element::i32) {
            ov::Tensor tensor{ov::element::i32, shape};
            auto* data = tensor.data<int32_t>();
            int32_t denom = 1;
            if (!shape.empty() && shape[0] != 0) {
                denom = static_cast<int32_t>(shape[0]);
            }
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = static_cast<int32_t>(i % denom);
            }
            inputs.insert({param, tensor});
        } else {
            FAIL() << "Unexpected parameter precision " << param->get_element_type();
        }
    }
}

TEST_P(StatefulSdpaBoolMaskTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    const auto inferencePrecision = GetParam();
    if (inferencePrecision == ov::element::bf16 && !ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP();
    }
    if (inferencePrecision == ov::element::f16) {
        if (!ov::with_cpu_x86_avx512_core_fp16() && !ov::with_cpu_neon_fp16()) {
            GTEST_SKIP();
        }
    }

    run();
    CheckPluginRelatedResults(compiledModel, "ScaledAttn");
}

}  // namespace test
}  // namespace ov
