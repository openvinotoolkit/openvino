// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/ctc_greedy_decoder.hpp"

#include <random>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/ctc_greedy_decoder.hpp"

namespace ov {
namespace test {
std::string CTCGreedyDecoderLayerTest::getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderParams>& obj) {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string targetDevice;
    bool merge_repeated;
    std::tie(model_type, shapes, merge_repeated, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';

    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "netPRC=" << model_type.get_type_name() << separator;
    result << "merge_repeated=" << std::boolalpha << merge_repeated << separator;
    result << "trgDev=" << targetDevice;

    return result.str();
}

void CTCGreedyDecoderLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    bool merge_repeated;
    std::tie(model_type, shapes, merge_repeated, targetDevice) = GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    size_t T = targetStaticShapes[0][0][0];
    size_t B = targetStaticShapes[0][0][1];

    std::mt19937 gen(1);
    std::uniform_int_distribution<unsigned long> dist(1, T);

    std::vector<int> sequence_mask_data(B * T, 0);
    for (int b = 0; b < B; b++) {
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
}  // namespace test
}  // namespace ov
