// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/nms_rotated.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/op/nms_rotated.hpp"

using namespace ov::test;

namespace LayerTestsDefinitions {

std::string NmsRotatedOpTest::getTestCaseName(const testing::TestParamInfo<NmsRotatedParams>& obj) {
    const auto& in_shapes = std::get<0>(obj.param);

    std::ostringstream result;

    result << "IS=(";
    for (size_t i = 0lu; i < in_shapes.size(); i++) {
        result << utils::partialShape2str({in_shapes[i].first}) << (i < in_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < in_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < in_shapes.size(); j++) {
            result << utils::vec2str(in_shapes[j].second[i]) << (j < in_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "_BoxPrc="    << std::get<1>(obj.param);
    result << "_MaxPrc="    << std::get<2>(obj.param);
    result << "_ThrPrc="    << std::get<3>(obj.param);
    result << "_OutPrc="    << std::get<4>(obj.param);
    result << "_MaxBox="    << std::get<5>(obj.param);
    result << "_IouThr="    << std::get<6>(obj.param);
    result << "_ScoreThr="  << std::get<7>(obj.param);
    result << "_SortDesc="  << utils::bool2str(std::get<8>(obj.param));
    result << "_Clockwise=" << utils::bool2str(std::get<9>(obj.param));
    result << "_ConstIn={"  << utils::bool2str(std::get<10>(obj.param)) << ","
                            << utils::bool2str(std::get<11>(obj.param)) << ","
                            << utils::bool2str(std::get<12>(obj.param)) << ","
                            << utils::bool2str(std::get<13>(obj.param)) << ","
                            << utils::bool2str(std::get<14>(obj.param)) << "}";

    const auto& config = std::get<15>(obj.param);
    if (!config.empty()) {
        result << "_Config={";
        for (const auto& conf_item : config) {
            result << "_" << conf_item.first << "=";
            conf_item.second.print(result);
        }
        result << "}";
    }

    result << "_Device=" << std::get<16>(obj.param);

    return result.str();
}

void NmsRotatedOpTest::SetUp() {
    const auto& params          = this->GetParam();
    const auto& in_shapes       = std::get<0>(params);
    const auto& boxes_prc       = std::get<1>(params);
    const auto& max_boxes_prc   = std::get<2>(params);
    const auto& thresholds_prc  = std::get<3>(params);
    const auto& out_prc         = std::get<4>(params);
    m_max_out_boxes_per_class   = std::get<5>(params);
    m_iou_threshold             = std::get<6>(params);
    m_score_threshold           = std::get<7>(params);
    const auto& sort_descending = std::get<8>(params);
    const auto& clockwise       = std::get<9>(params);
    const auto& is_0_in_const   = std::get<10>(params);
    const auto& is_1_in_const   = std::get<11>(params);
    const auto& is_2_in_const   = std::get<12>(params);
    const auto& is_3_in_const   = std::get<13>(params);
    const auto& is_4_in_const   = std::get<14>(params);
    configuration               = std::get<15>(params);
    targetDevice                = std::get<16>(params);

    std::vector<InputShape> actual_shapes;
    ov::ParameterVector in_params;
    std::vector<std::shared_ptr<ov::Node>> inputs;
    const auto in_shape_1d = InputShape{{1}, {{1}}};

#define CONST_CASE(P, S, H, L)                                                                                          \
    case P: {                                                                                                           \
        auto start_from = ov::element_type_traits<P>::value_type(L);                                                    \
        auto range = ov::element_type_traits<P>::value_type(H) - start_from;                                            \
        inputs.push_back(ov::test::utils::make_constant(P, S, ov::test::utils::InputGenerateData(start_from, range)));  \
        break; }

#define CREATE_INPUT(C, P, S, N, H, L)                                                                                     \
    if (C) {                                                                                                               \
        switch (P) {                                                                                                       \
            CONST_CASE(ElementType::f32,  S.second[0], H, L)                                                               \
            CONST_CASE(ElementType::f16,  S.second[0], H, L)                                                               \
            CONST_CASE(ElementType::bf16, S.second[0], H, L)                                                               \
            CONST_CASE(ElementType::i32,  S.second[0], H, L)                                                               \
            CONST_CASE(ElementType::i64,  S.second[0], H, L)                                                               \
            default: OPENVINO_THROW("NmsRotated does not support precision ", P, " for the ", N, " input.");               \
        }                                                                                                                  \
    } else {                                                                                                               \
        actual_shapes.push_back(S);                                                                                        \
        if (S.first.rank() == 0) {                                                                                         \
            in_params.push_back(std::make_shared<ov::op::v0::Parameter>(P, S.second.front()));                             \
        } else {                                                                                                           \
            in_params.push_back(std::make_shared<ov::op::v0::Parameter>(P, S.first));                                      \
        }                                                                                                                  \
        in_params.back()->set_friendly_name(N);                                                                            \
        inputs.push_back(in_params.back());                                                                                \
    }

    CREATE_INPUT(is_0_in_const, boxes_prc,      in_shapes[0], "Boxes", 30, 10)
    CREATE_INPUT(is_1_in_const, boxes_prc,      in_shapes[1], "Scores", 1, 0)
    CREATE_INPUT(is_2_in_const, max_boxes_prc,  in_shape_1d, "MaxOutputBoxesPerClass", m_max_out_boxes_per_class, m_max_out_boxes_per_class)
    CREATE_INPUT(is_3_in_const, thresholds_prc, in_shape_1d, "IouThreshold", m_iou_threshold, m_iou_threshold)
    CREATE_INPUT(is_4_in_const, thresholds_prc, in_shape_1d, "ScoreThreshold", m_score_threshold, m_score_threshold)

#undef CONST_CASE
#undef CREATE_INPUT

    init_input_shapes(actual_shapes);

    const auto nms_op = std::make_shared<ov::op::v13::NMSRotated>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                                                                    sort_descending, out_prc, clockwise);
    ov::ResultVector results;
    for (size_t i = 0lu; i < nms_op->get_output_size(); i++) {
        results.push_back(std::make_shared<ov::op::v0::Result>(nms_op->output(i)));
    }

    function = std::make_shared<ov::Model>(results, in_params, "NMSRotated");
}

template<typename TD, typename TS>
void fill_data(TD* dst, const TS* src, size_t len) {
    for (size_t i = 0llu; i < len; i++) {
        dst[i] = static_cast<TD>(src[i]);
    }
}

void NmsRotatedOpTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();

    for (size_t i = 0llu; i < func_inputs.size(); ++i) {
        const auto& func_input = func_inputs[i];
        const auto& name = func_input.get_node()->get_friendly_name();
        const auto& in_prc = func_input.get_element_type();
        auto tensor = ov::Tensor(in_prc, targetInputStaticShapes[i]);

#define FILL_DATA(P, S, L)                                                          \
case P :                                                                            \
fill_data(tensor.data<ov::element_type_traits<P>::value_type>(), S, L); break;

#define GEN_DATA(P, R, S, K)                                                                                                               \
case P :                                                                                                                                   \
utils::fill_data_random(tensor.data<ov::element_type_traits<P>::value_type>(), shape_size(targetInputStaticShapes[i]), R, S, K); break;

        if (name == "Boxes") {
            switch (in_prc) {
                GEN_DATA(ElementType::f32, 30, 20, 1)
                GEN_DATA(ElementType::f16, 30, 20, 1)
                GEN_DATA(ElementType::bf16, 30, 20, 1)
                default:
                    OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the Scores input.");
            }
        } else if (name == "Scores") {
            switch (in_prc) {
                GEN_DATA(ElementType::f32, 1, 0, 100)
                GEN_DATA(ElementType::f16, 1, 0, 100)
                GEN_DATA(ElementType::bf16, 1, 0, 100)
                default:
                    OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the Scores input.");
            }
        } else if (name == "MaxOutputBoxesPerClass") {
            switch (in_prc) {
                FILL_DATA(ElementType::i64, &m_max_out_boxes_per_class, 1)
                FILL_DATA(ElementType::i32, &m_max_out_boxes_per_class, 1)
                default:
                    OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the MaxOutputBoxesPerClass input.");
            }
        } else if (name == "IouThreshold") {
            switch (in_prc) {
                FILL_DATA(ElementType::f32,  &m_iou_threshold, 1)
                FILL_DATA(ElementType::f16,  &m_iou_threshold, 1)
                FILL_DATA(ElementType::bf16, &m_iou_threshold, 1)
                default:
                    OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the IouThreshold input.");
            }
        } else if (name == "ScoreThreshold") {
            switch (in_prc) {
                FILL_DATA(ElementType::f32,  &m_score_threshold, 1)
                FILL_DATA(ElementType::f16,  &m_score_threshold, 1)
                FILL_DATA(ElementType::bf16, &m_score_threshold, 1)
                default:
                    OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the ScoreThreshold input.");
            }
        }

#undef GEN_DATA
#undef FILL_DATA

        inputs.insert({func_input.get_node_shared_ptr(), tensor});
    }
}

} // namespace LayerTestsDefinitions
