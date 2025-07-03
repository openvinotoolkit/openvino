// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/embedding_bag_offsets_sum.hpp"
#include "common_test_utils/node_builders/embedding_bag_offsets_sum.hpp"

namespace ov {
namespace test {
std::string EmbeddingBagOffsetsSumLayerTest::getTestCaseName(const testing::TestParamInfo<embeddingBagOffsetsSumLayerTestParamsSet>& obj) {
    embeddingBagOffsetsSumParams emb_params;
    ov::element::Type model_type, ind_type;
    std::vector<InputShape> shapes;
    std::string target_device;
    std::tie(emb_params, shapes, model_type, ind_type, target_device) = obj.param;
    std::vector<size_t> indices, offsets;
    size_t default_index;
    bool with_weights, with_def_index;
    std::tie(indices, offsets, default_index, with_weights, with_def_index) = emb_params;

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
    result << "I" << ov::test::utils::vec2str(indices) << "_";
    result << "O" << ov::test::utils::vec2str(offsets) << "_";
    result << "DI" << default_index << "_";
    result << "WW" << with_weights << "_";
    result << "WDI" << with_def_index << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "indPRC=" << ind_type.get_type_name() << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void EmbeddingBagOffsetsSumLayerTest::SetUp() {
    embeddingBagOffsetsSumParams emb_params;
    ov::element::Type model_type, ind_type;
    std::vector<InputShape> shapes;
    std::tie(emb_params, shapes, model_type, ind_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    std::vector<size_t> indices, offsets;
    bool with_weights, with_def_index;
    size_t default_index;
    std::tie(indices, offsets, default_index, with_weights, with_def_index) = emb_params;

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto embBag = ov::test::utils::make_embedding_bag_offsets_sum(model_type, ind_type, param, indices, offsets, default_index, with_weights, with_def_index);

    auto result = std::make_shared<ov::op::v0::Result>(embBag);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "embeddingBagOffsetsSum");
}
}  // namespace test
}  // namespace ov
