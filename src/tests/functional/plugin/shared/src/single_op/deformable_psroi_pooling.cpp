// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/deformable_psroi_pooling.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/deformable_psroi_pooling.hpp"

namespace ov {
namespace test {

std::string DeformablePSROIPoolingLayerTest::getTestCaseName(const testing::TestParamInfo<deformablePSROILayerTestParams>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    int64_t outputDim;
    int64_t groupSize;
    float spatialScale;
    std::vector<int64_t> spatialBinsXY;
    float trans_std;
    int64_t part_size;
    std::string target_device;
    deformablePSROISpecificParams opParams;

    std::tie(opParams, shapes, model_type, target_device) = obj.param;
    std::tie(outputDim, groupSize, spatialScale, spatialBinsXY,
    trans_std, part_size) = opParams;

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
    result << "out_dim=" << outputDim << "_";
    result << "group_size=" << groupSize << "_";
    result << "scale=" << spatialScale << "_";
    result << "bins_x=" << spatialBinsXY[0] << "_";
    result << "bins_y=" << spatialBinsXY[1] << "_";
    result << "trans_std=" << trans_std << "_";
    result << "part_size=" << part_size << "_";
    result << "prec=" << model_type.get_type_name() << "_";
    result << "dev=" << target_device;
    return result.str();
}

void DeformablePSROIPoolingLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    int64_t outputDim;
    int64_t groupSize;
    std::string mode = "bilinear_deformable";
    std::vector<int64_t> spatialBinsXY;
    float trans_std, spatial_scale;
    int64_t part_size;
    deformablePSROISpecificParams opParams;

    std::tie(opParams, shapes, model_type, targetDevice) = this->GetParam();
    std::tie(outputDim, groupSize, spatial_scale, spatialBinsXY, trans_std, part_size) = opParams;
    init_input_shapes(shapes);

    ov::ParameterVector params;
    std::shared_ptr<ov::op::v1::DeformablePSROIPooling> defomablePSROIPooling;

    if (2 == inputDynamicShapes.size()) { // Test without optional third input (offsets)
        params = ov::ParameterVector{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                     std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1])};

        defomablePSROIPooling = std::make_shared<ov::op::v1::DeformablePSROIPooling>(params[0],
                                                                                     params[1],
                                                                                     outputDim,
                                                                                     spatial_scale,
                                                                                     groupSize,
                                                                                     mode,
                                                                                     spatialBinsXY[0],
                                                                                     spatialBinsXY[1],
                                                                                     trans_std,
                                                                                     part_size);
    } else {
        params = ov::ParameterVector{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                     std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
                                     std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};

        defomablePSROIPooling = std::make_shared<ov::op::v1::DeformablePSROIPooling>(params[0],
                                                                                     params[1],
                                                                                     params[2],
                                                                                     outputDim,
                                                                                     spatial_scale,
                                                                                     groupSize,
                                                                                     mode,
                                                                                     spatialBinsXY[0],
                                                                                     spatialBinsXY[1],
                                                                                     trans_std,
                                                                                     part_size);
    }

    auto result = std::make_shared<ov::op::v0::Result>(defomablePSROIPooling);
    function = std::make_shared<ov::Model>(result, params, "deformable_psroi_pooling");
}
}  // namespace test
}  // namespace ov
