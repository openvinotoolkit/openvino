// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <tuple>
#include <random>

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
    ov::element::Type,          // Model type
    InputShape,                 // Input shape
    bool,                       // Merge repeated
    std::string                 // Device name
> ctcGreedyDecoderParams;

class CTCGreedyDecoderLayerGPUTest
    :  public testing::WithParamInterface<ctcGreedyDecoderParams>,
       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderParams>& obj) {
        ov::element::Type model_type;
        InputShape input_shape;
        std::string targetDevice;
        bool merge_repeated;
        std::tie(model_type, input_shape, merge_repeated, targetDevice) = obj.param;

        std::ostringstream result;
        const char separator = '_';

        result << "IS=(";
        result << ov::test::utils::partialShape2str({input_shape.first}) << "_" << "TS=(";
        for (size_t i = 0lu; i < input_shape.second.size(); i++) {
            result << ov::test::utils::vec2str(input_shape.second[i]) << "_";
        }
        result << ")_";
        result << "netPRC=" << model_type.get_type_name() << separator;
        result << "merge_repeated=" << std::boolalpha << merge_repeated << separator;
        result << "trgDev=" << targetDevice;

        return result.str();
    }
protected:
    void SetUp() override {
        ov::element::Type model_type;
        InputShape input_shape;
        bool merge_repeated;
        std::tie(model_type, input_shape, merge_repeated, targetDevice) = GetParam();
        inputDynamicShapes = {input_shape.first, {}};
        for (size_t i = 0; i < input_shape.second.size(); ++i) {
            targetStaticShapes.push_back({input_shape.second[i], {}});
        }

        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

        size_t T = targetStaticShapes[0][0][0];
        size_t B = targetStaticShapes[0][0][1];

        std::mt19937 gen(1);
        std::uniform_int_distribution<unsigned long> dist(1, T);

        std::vector<int> sequence_mask_data(B * T, 0);
        for (size_t b = 0; b < B; b++) {
            int len = dist(gen);
            for (int t = 0; t < len; t++) {
                sequence_mask_data[t * B + b] = 1;
            }
        }
        auto sequence_mask_node = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{T, B}, sequence_mask_data);

        auto ctc_greedy_decoder = std::make_shared<ov::op::v0::CTCGreedyDecoder>(param, sequence_mask_node, merge_repeated);

        auto result = std::make_shared<ov::op::v0::Result>(ctc_greedy_decoder);
        function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "CTCGreedyDecoder");
    }
};


TEST_P(CTCGreedyDecoderLayerGPUTest, Inference) {
    run();
};

// Common params
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};
std::vector<bool> mergeRepeated{true, false};

std::vector<ov::test::InputShape> input_shapes_dynamic = {
    {
        {{-1, -1, -1}, {{ 50, 3, 3 }}},
        {{-1, -1, -1}, {{ 50, 3, 7 }}},
        {{-1, -1, -1}, {{ 50, 3, 8 }}},
        {{-1, -1, -1}, {{ 50, 3, 16 }}},
        {{-1, -1, -1}, {{ 50, 3, 128 }}},
        {{-1, -1, -1}, {{ 50, 3, 49 }}},
        {{-1, -1, -1}, {{ 50, 3, 55 }}},
        {{-1, -1, -1}, {{ 1, 1, 16 }}}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_CtcGreedyDecoderBasicDynamic,
                         CTCGreedyDecoderLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(input_shapes_dynamic),
                                            ::testing::ValuesIn(mergeRepeated),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         CTCGreedyDecoderLayerGPUTest::getTestCaseName);
}  // namespace
