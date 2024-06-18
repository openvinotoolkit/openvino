// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {
namespace GPULayerTestsDefinitions {

typedef std::tuple<
    std::vector<ov::test::InputShape>,  // Input shapes
    ov::test::ElementType,              // Boxes and scores input precisions
    ov::test::ElementType,              // Max output boxes input precisions
    ov::test::ElementType,              // Thresholds precisions
    ov::test::ElementType,              // Output type
    int64_t,                            // Max output boxes per class
    float,                              // IOU threshold
    float,                              // Score threshold
    bool,                               // Sort result descending
    bool,                               // Clockwise
    bool,                               // Is 1st input constant
    bool,                               // Is 2nd input constant
    bool,                               // Is 3rd input constant
    bool,                               // Is 4th input constant
    bool,                               // Is 5th input constant
    ov::AnyMap,                         // Additional configuration
    std::string                         // Device name
> NmsRotatedParamsGPU;

class NmsRotatedLayerTestGPU : public testing::WithParamInterface<NmsRotatedParamsGPU>,
                         public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsRotatedParamsGPU>& obj);

protected:
    void SetUp() override;
    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override;
    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;

private:
    int64_t m_max_out_boxes_per_class;
    float m_iou_threshold;
    float m_score_threshold;
};

TEST_P(NmsRotatedLayerTestGPU, CompareWithRefs) {
    run();
};

std::string NmsRotatedLayerTestGPU::getTestCaseName(const testing::TestParamInfo<NmsRotatedParamsGPU>& obj) {
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

void NmsRotatedLayerTestGPU::SetUp() {
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

void NmsRotatedLayerTestGPU::compare(const std::vector<ov::Tensor> &expectedOutputs, const std::vector<ov::Tensor> &actualOutputs) {
    struct OutBox {
        OutBox() = default;

        OutBox(int32_t batchId, int32_t classId, int32_t boxId, float score) {
            this->batchId = batchId;
            this->classId = classId;
            this->boxId = boxId;
            this->score = score;
        }

        bool operator==(const OutBox& rhs) const {
            return batchId == rhs.batchId && classId == rhs.classId && boxId == rhs.boxId;
        }

        int32_t batchId;
        int32_t classId;
        int32_t boxId;
        float score;
    };

    std::vector<OutBox> expected;
    {
        const auto selected_indices_size = expectedOutputs[0].get_size();
        const auto selected_scores_size = expectedOutputs[1].get_size();

        ASSERT_EQ(selected_indices_size, selected_scores_size);

        const auto boxes_count = selected_indices_size / 3;
        expected.resize(boxes_count);

        if (expectedOutputs[0].get_element_type().bitwidth() == 32) {
            auto selected_indices_data = reinterpret_cast<const int32_t*>(expectedOutputs[0].data());

            for (size_t i = 0; i < selected_indices_size; i += 3) {
                expected[i / 3].batchId = selected_indices_data[i + 0];
                expected[i / 3].classId = selected_indices_data[i + 1];
                expected[i / 3].boxId = selected_indices_data[i + 2];
            }
        } else {
            auto selected_indices_data = reinterpret_cast<const int64_t*>(expectedOutputs[0].data());

            for (size_t i = 0; i < selected_indices_size; i += 3) {
                expected[i / 3].batchId = static_cast<int32_t>(selected_indices_data[i + 0]);
                expected[i / 3].classId = static_cast<int32_t>(selected_indices_data[i + 1]);
                expected[i / 3].boxId = static_cast<int32_t>(selected_indices_data[i + 2]);
            }
        }

        if (expectedOutputs[1].get_element_type().bitwidth() == 32) {
            auto selected_scores_data = reinterpret_cast<const float*>(expectedOutputs[1].data());
            for (size_t i = 0; i < selected_scores_size; i += 3) {
                expected[i / 3].score = selected_scores_data[i + 2];
            }
        } else {
            auto selected_scores_data = reinterpret_cast<const double*>(expectedOutputs[1].data());
            for (size_t i = 0; i < selected_scores_size; i += 3) {
                expected[i / 3].score = static_cast<float>(selected_scores_data[i + 2]);
            }
        }
    }

    std::vector<OutBox> actual;
    {
        const auto selected_indices_size = actualOutputs[0].get_size();
        const auto selected_scores_data = reinterpret_cast<const float*>(actualOutputs[1].data());
        if (actualOutputs[0].get_element_type().bitwidth() == 32) {
            const auto selected_indices_data = reinterpret_cast<const int32_t*>(actualOutputs[0].data());
            for (size_t i = 0; i < selected_indices_size; i += 3) {
                const int32_t batchId = selected_indices_data[i + 0];
                const int32_t classId = selected_indices_data[i + 1];
                const int32_t boxId = selected_indices_data[i + 2];
                const float score = selected_scores_data[i + 2];
                if (batchId == -1 || classId == -1 || boxId == -1)
                    break;

                actual.emplace_back(batchId, classId, boxId, score);
            }
        } else {
            const auto selected_indices_data = reinterpret_cast<const int64_t*>(actualOutputs[0].data());
            for (size_t i = 0; i < selected_indices_size; i += 3) {
                const int64_t batchId = selected_indices_data[i + 0];
                const int64_t classId = selected_indices_data[i + 1];
                const int64_t boxId = selected_indices_data[i + 2];
                const float score = selected_scores_data[i + 2];
                if (batchId == -1 || classId == -1 || boxId == -1)
                    break;

                actual.emplace_back(batchId, classId, boxId, score);
            }
        }
    }

    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        abs_threshold = ov::test::utils::get_eps_by_ov_type(ov::element::f32) * expected[i].score;
        ASSERT_EQ(expected[i], actual[i]) << ", i=" << i;
        ASSERT_NEAR(expected[i].score, actual[i].score, abs_threshold) << ", i=" << i;
    }
}

template<typename TD, typename TS>
void fill_data(TD* dst, const TS* src, size_t len) {
    for (size_t i = 0llu; i < len; i++) {
        dst[i] = static_cast<TD>(src[i]);
    }
}

void NmsRotatedLayerTestGPU::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
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
                default:
                    OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the Scores input.");
            }
        } else if (name == "Scores") {
            switch (in_prc) {
                GEN_DATA(ElementType::f32, 1, 0, 100)
                GEN_DATA(ElementType::f16, 1, 0, 100)
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
                default:
                    OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the IouThreshold input.");
            }
        } else if (name == "ScoreThreshold") {
            switch (in_prc) {
                FILL_DATA(ElementType::f32,  &m_score_threshold, 1)
                FILL_DATA(ElementType::f16,  &m_score_threshold, 1)
                default:
                    OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the ScoreThreshold input.");
            }
        }

#undef GEN_DATA
#undef FILL_DATA

        inputs.insert({func_input.get_node_shared_ptr(), tensor});
    }
}

namespace {
const std::vector<std::vector<ov::test::InputShape>> inShapeParams = {
    {
        { {}, {{2, 50, 5}} },
        { {}, {{2, 50, 50}} }
    },
    {
        { {}, {{9, 10, 5}} },
        { {}, {{9, 10, 10}} }
    }
};

const std::vector<ov::element::Type_t> outType = {ov::element::i32, ov::element::i64};
const std::vector<ov::element::Type_t> inputPrecisions = {ov::element::f32, ov::element::f16};
const ov::AnyMap empty_plugin_config{};

INSTANTIATE_TEST_SUITE_P(smoke_NmsRotatedLayerTest,
                         NmsRotatedLayerTestGPU,
                         ::testing::Combine(::testing::ValuesIn(inShapeParams),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(ov::element::i32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(outType),
                                            ::testing::Values(5, 20),
                                            ::testing::Values(0.3f, 0.7f),
                                            ::testing::Values(0.3f, 0.7f),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         NmsRotatedLayerTestGPU::getTestCaseName);

}  // namespace
}  // namespace GPULayerTestsDefinitions
} // namespace test
} // namespace ov
