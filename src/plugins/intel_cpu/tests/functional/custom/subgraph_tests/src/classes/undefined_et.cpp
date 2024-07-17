// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Modes: ACCURACY; PERFORMANCE + INFERENCE_PRECISION->undefined
// Expected: original execution type from the IR.
//
//       -----------
//      | Parameter |
//       -----------
//            | f32/f16/bf16
//    -----------------
//   |  RandomUniform  | supports execution in f32/f16/bf16 on x86 and ARM
//    -----------------
//            |
//       -----------
//      |  Convert  |
//       -----------
//            | f32
//       -----------
//      |  Eltwise  |
//       -----------
//            |
//       -----------
//      |   Result  |
//       -----------

#include "custom/subgraph_tests/include/undefined_et.hpp"
#include "utils/precision_support.h"

namespace ov {
namespace test {

std::string UndefinedEtSubgraphTest::getTestCaseName(const testing::TestParamInfo<UndefinedEtCpuParams>& obj) {
    std::ostringstream result;

    result << "DataET=" << std::get<0>(obj.param);

    const auto& config = std::get<1>(obj.param);
    if (!config.empty()) {
        result << "_PluginConf={";
        for (const auto& conf_item : config) {
            result << "_" << conf_item.first << "=";
            conf_item.second.print(result);
            result << "_";
        }
        result << "}";
    }

    return result.str();
}

void UndefinedEtSubgraphTest::SetUp() {
    targetDevice = test::utils::DEVICE_CPU;

    const auto& params = this->GetParam();
    m_data_et = std::get<0>(params);
    const auto& config = std::get<1>(params);

    configuration.insert(config.begin(), config.end());
    auto it = configuration.find(hint::execution_mode.name());
    ASSERT_NE(configuration.end(), it);
    m_mode = it->second.as<hint::ExecutionMode>();

    init_input_shapes({ {{}, {{3}}}, {{}, {{1}}}, {{}, {{1}}} });
    auto param_0 = std::make_shared<op::v0::Parameter>(element::i64, inputDynamicShapes[0]);
    param_0->set_friendly_name("shape");
    auto param_1 = std::make_shared<op::v0::Parameter>(m_data_et, inputDynamicShapes[1]);
    param_1->set_friendly_name("minval");
    auto param_2 = std::make_shared<op::v0::Parameter>(m_data_et, inputDynamicShapes[2]);
    param_2->set_friendly_name("maxval");

    auto rnd_unfm = std::make_shared<op::v8::RandomUniform>(param_0, param_1, param_2, m_data_et, 1lu, 0lu);
    auto cvt_f32 = std::make_shared<op::v0::Convert>(rnd_unfm, element::f32);
    auto logical_not = std::make_shared<op::v1::LogicalNot>(cvt_f32);

    function = std::make_shared<ov::Model>(OutputVector{logical_not->output(0)}, ParameterVector{param_0, param_1, param_2}, "UndefinedET");

    // TODO: Need to remove when the hardware checking for f16 will be eliminated in the Transformations pipeline.
    if (m_data_et == element::f16 && !ov::intel_cpu::hasHardwareSupport(m_data_et)) {
        abs_threshold = 1.f;
        rel_threshold = 0.1f;
    }
}

template<typename TD, typename TS>
void fill_data(TD* dst, const TS* src, size_t len) {
    for (size_t i = 0llu; i < len; i++) {
        dst[i] = static_cast<TD>(src[i]);
    }
}

void UndefinedEtSubgraphTest::generate_inputs(const std::vector<ov::Shape>& target_shapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();

#define ET_CASE(P, S, L)                                                           \
case P :                                                                           \
fill_data(tensor->data<ov::element_type_traits<P>::value_type>(), S, L); break;

    for (size_t i = 0lu; i < func_inputs.size(); i++) {
        const auto& param = func_inputs[i];
        const auto& name = param.get_node()->get_friendly_name();
        const auto& in_prc = param.get_element_type();
        std::shared_ptr<ov::Tensor> tensor;

        if (name == "shape") {
            static const int64_t shape[] = {3, 5, 7};
            tensor = std::make_shared<ov::Tensor>(in_prc, Shape{3});
            switch (in_prc) {
                ET_CASE(element::i32, shape, 3)
                ET_CASE(element::i64, shape, 3)
                default:
                    OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Shape input.");
            }
        } else if (name == "minval") {
            static const float min_val = 0.f;
            tensor = std::make_shared<ov::Tensor>(in_prc, target_shapes[i]);
            switch (in_prc) {
                ET_CASE(ElementType::f32,  &min_val, 1)
                ET_CASE(ElementType::f16,  &min_val, 1)
                ET_CASE(ElementType::bf16, &min_val, 1)
                default:
                    OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Minval input.");
            }
        } else if (name == "maxval") {
            static const float max_val = 20.f;
            tensor = std::make_shared<ov::Tensor>(in_prc, target_shapes[i]);
            switch (in_prc) {
                ET_CASE(ElementType::f32,  &max_val, 1)
                ET_CASE(ElementType::f16,  &max_val, 1)
                ET_CASE(ElementType::bf16, &max_val, 1)
                default:
                    OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Maxval input.");
            }
        }

#undef ET_CASE

        inputs.insert({param.get_node_shared_ptr(), *tensor});
    }
}

TEST_P(UndefinedEtSubgraphTest, CompareWithRefs) {
    run();

    if (IsSkipped()) {
        return;
    }

    ASSERT_EQ(compiledModel.get_property(ov::hint::execution_mode), m_mode);
    ASSERT_EQ(compiledModel.get_property(ov::hint::inference_precision), element::undefined);

    size_t rnd_unfm_counter = 0lu;
    size_t logical_not_counter = 0lu;
    auto expected_dt = m_data_et;
    if (!ov::intel_cpu::hasHardwareSupport(expected_dt)) {
        expected_dt = element::f32;
    }
    for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
        auto rt_info = node->get_rt_info();
        auto it = rt_info.find(exec_model_info::LAYER_TYPE);
        ASSERT_NE(rt_info.end(), it);
        auto op_name = it->second.as<std::string>();

        if (op_name == "RandomUniform") {
            ASSERT_EQ(node->get_output_element_type(0), expected_dt);
            rnd_unfm_counter++;
        }
        if (op_name == "Eltwise") {
            ASSERT_EQ(node->get_output_element_type(0), element::f32);
            logical_not_counter++;
        }
    }
    ASSERT_EQ(rnd_unfm_counter, 1lu);
    ASSERT_EQ(logical_not_counter, 1lu);
};

}  // namespace test
}  // namespace ov
