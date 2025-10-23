// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/ctc_greedy_decoder_seq_len.hpp"

#include <string>
#include <vector>
#include <memory>
#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

namespace ov {
namespace test {
std::string CTCGreedyDecoderSeqLenLayerTest::getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderSeqLenParams>& obj) {
    const auto& [shapes, sequenceLengths, dataPrecision, indicesPrecision, blankIndex, mergeRepeated, targetDevice] =
        obj.param;

    std::ostringstream result;

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
    result << "seqLen=" << sequenceLengths << '_';
    result << "dataPRC=" << dataPrecision.get_type_name() << '_';
    result << "idxPRC=" << indicesPrecision.get_type_name() << '_';
    result << "BlankIdx=" << blankIndex << '_';
    result << "mergeRepeated=" << std::boolalpha << mergeRepeated << '_';
    result << "trgDev=" << targetDevice;

    return result.str();
}

void CTCGreedyDecoderSeqLenLayerTest::SetUp() {
    const auto& [shapes, sequenceLengths, model_type, indices_type, _blankIndex, mergeRepeated, _targetDevice] =
        GetParam();
    targetDevice = _targetDevice;
    auto blankIndex = _blankIndex;
    init_input_shapes(shapes);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};

    const auto sequenceLenNode = [&, sequenceLengths = sequenceLengths, indices_type = indices_type] {
        const size_t B = targetStaticShapes[0][0][0];
        const size_t T = targetStaticShapes[0][0][1];

        // Cap sequence length up to T
        const int seqLen = std::min<int>(T, sequenceLengths);

        std::mt19937 gen{42};
        std::uniform_int_distribution<int> dist(1, seqLen);

        std::vector<int> sequenceLenData(B);
        for (int b = 0; b < B; b++) {
            const int len = dist(gen);
            sequenceLenData[b] = len;
        }

        return std::make_shared<ov::op::v0::Constant>(indices_type, ov::Shape{B}, sequenceLenData);
    }();

    // Cap blank index up to C - 1
    int C = targetStaticShapes[0][0][2];
    blankIndex = std::min(blankIndex, C - 1);

    const auto blankIndexNode = [&, indices_type = indices_type] {
        if (indices_type == element::i32) {
            const auto blankIdxDataI32 = std::vector<int32_t>{blankIndex};
            return std::make_shared<ov::op::v0::Constant>(indices_type, ov::Shape{1}, blankIdxDataI32);
        } else if (indices_type == element::i64) {
            const auto blankIdxDataI64 = std::vector<int64_t>{blankIndex};
            return std::make_shared<ov::op::v0::Constant>(indices_type, ov::Shape{1}, blankIdxDataI64);
        }
        throw std::logic_error("Unsupported index precision");
    }();

    auto ctcGreedyDecoderSeqLen = std::make_shared<ov::op::v6::CTCGreedyDecoderSeqLen>(params[0],
                                                                                       sequenceLenNode,
                                                                                       blankIndexNode,
                                                                                       mergeRepeated,
                                                                                       indices_type,
                                                                                       indices_type);

    ov::OutputVector results;
    for (int i = 0; i < ctcGreedyDecoderSeqLen->get_output_size(); i++) {
        results.push_back(std::make_shared<ov::op::v0::Result>(ctcGreedyDecoderSeqLen->output(i)));
    }
    function = std::make_shared<ov::Model>(results, params, "CTCGreedyDecoderSeqLen");
}
}  // namespace test
}  // namespace ov
