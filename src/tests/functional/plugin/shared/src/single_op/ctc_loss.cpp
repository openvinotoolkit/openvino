// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/ctc_loss.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/ctc_loss.hpp"

namespace ov {
namespace test {
std::string CTCLossLayerTest::getTestCaseName(const testing::TestParamInfo<CTCLossParams>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type fp_type, int_type;
    bool preprocess_collapse_repeated, ctc_merge_repeated, unique;
    std::vector<int> logits_length, labels_length;
    std::vector<std::vector<int>> labels;
    int blank_index;
    std::string targetDevice;
    CTCLossParamsSubset ctcLossArgsSubset;
    std::tie(ctcLossArgsSubset, shapes, fp_type, int_type, targetDevice) = obj.param;
    std::tie(logits_length, labels, labels_length, blank_index, preprocess_collapse_repeated, ctc_merge_repeated, unique) = ctcLossArgsSubset;

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
    result << "LL=" << ov::test::utils::vec2str(logits_length) << "_";
    result << "A=" << ov::test::utils::vec2str(labels) << "_";
    result << "AL=" << ov::test::utils::vec2str(labels_length) << "_";
    result << "BI=" << blank_index << "_";
    result << "PCR=" << preprocess_collapse_repeated << "_";
    result << "CMR=" << ctc_merge_repeated << "_";
    result << "U=" << unique << "_";
    result << "PF=" << fp_type.get_type_name() << "_";
    result << "PI=" << int_type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void CTCLossLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type fp_type, int_type;
    bool preprocess_collapse_repeated, ctc_merge_repeated, unique;
    std::vector<int> logits_length, labels_length;
    std::vector<std::vector<int>> labels;
    int blank_index;
    CTCLossParamsSubset ctcLossArgsSubset;
    std::tie(ctcLossArgsSubset, shapes, fp_type, int_type, targetDevice) = this->GetParam();
    std::tie(logits_length, labels, labels_length, blank_index, preprocess_collapse_repeated,
        ctc_merge_repeated, unique) = ctcLossArgsSubset;
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(fp_type, inputDynamicShapes.front());

    size_t N = targetStaticShapes[0][0][0];
    size_t T = targetStaticShapes[0][0][1];

    std::vector<int> labelsOneD(N * T);
    for (int i = 0; i < labels.size(); i++)
        std::copy(labels[i].begin(), labels[i].end(), labelsOneD.data() + i * T);

    auto logits_length_node = std::make_shared<ov::op::v0::Constant>(int_type, ov::Shape{N}, logits_length);
    auto labels_node = std::make_shared<ov::op::v0::Constant>(int_type, ov::Shape{N, T}, labelsOneD);
    auto labels_length_node = std::make_shared<ov::op::v0::Constant>(int_type, ov::Shape{N}, labels_length);
    auto blank_index_node = std::make_shared<ov::op::v0::Constant>(int_type, ov::Shape{}, blank_index);

    auto ctc_loss = std::make_shared<ov::op::v4::CTCLoss>(param,
                                                          logits_length_node,
                                                          labels_node,
                                                          labels_length_node,
                                                          blank_index_node,
                                                          preprocess_collapse_repeated,
                                                          ctc_merge_repeated,
                                                          unique);

    auto result = std::make_shared<ov::op::v0::Result>(ctc_loss);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "CTCLoss");

    if (fp_type == ov::element::f16) {
        abs_threshold = 9e-3;
    }
}
}  // namespace test
}  // namespace ov
