// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"

#include "vpu/private_plugin_config.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <functional_test_utils/blob_utils.hpp>
#include <precision_utils.h>
#include <ngraph/opsets/opset3.hpp>

#include <tuple>
#include <vector>
#include <string>
#include <memory>

using InputShape = InferenceEngine::SizeVector;
using ShapeDescriptor = std::vector<int>;

using OutShapeOfReshapeParam = std::tuple<
        InputShape,       // Input shape
        ShapeDescriptor,  // out shape descriptor
        bool>;            // Special zero

using OutShapeOfReshapeTestParam = std::tuple<
        OutShapeOfReshapeParam,          // Shape params
        LayerTestsUtils::TargetDevice>;  // Device name


namespace LayerTestsDefinitions {

class OutShapeOfReshapeLayerTest : public testing::WithParamInterface<OutShapeOfReshapeTestParam>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<OutShapeOfReshapeTestParam>& obj) {
        OutShapeOfReshapeParam shapesParam;
        std::string targetDevice;
        std::tie(shapesParam, targetDevice) = obj.param;

        const auto& inputShape = std::get<0>(shapesParam);
        const auto& outShapeDescriptor = std::get<1>(shapesParam);
        const auto& specialZero = std::get<2>(shapesParam);

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "OSD=" << CommonTestUtils::vec2str(outShapeDescriptor) << "_";
        result << "SZ=" << std::to_string(specialZero) << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
        configuration[InferenceEngine::MYRIAD_DISABLE_REORDER] = CONFIG_VALUE(YES);

        OutShapeOfReshapeParam shapesParam;
        std::tie(shapesParam, targetDevice) = this->GetParam();
        inPrc = InferenceEngine::Precision::I32;
        outPrc = InferenceEngine::Precision::I32;

        const auto& inputShape = std::get<0>(shapesParam);
        const auto& outShapeDescriptor = std::get<1>(shapesParam);
        const auto& specialZero = std::get<2>(shapesParam);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        const auto inputShapeParam = std::make_shared<ngraph::opset3::Parameter>(
                ngPrc, ngraph::Shape{inputShape.size()});
        const auto outShapeDescriptorConst = std::make_shared<ngraph::opset3::Constant>(
                ngPrc, ngraph::Shape{outShapeDescriptor.size()}, outShapeDescriptor);

        const auto outShapeOfReshape = std::make_shared<ngraph::vpu::op::OutShapeOfReshape>(
                inputShapeParam, outShapeDescriptorConst, specialZero);
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(outShapeOfReshape)};
        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{inputShapeParam});
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        OutShapeOfReshapeParam shapesParam;
        std::string targetDevice;
        std::tie(shapesParam, targetDevice) = this->GetParam();
        const auto& inputShape = std::get<0>(shapesParam);

        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto dataPtr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rwmap().as<int32_t*>();
        for (size_t i = 0; i < blob->size(); ++i) {
            dataPtr[i] = inputShape[i];
        }

        return blob;
    }
};

TEST_P(OutShapeOfReshapeLayerTest, accuracy) {
    Run();
}

std::vector<OutShapeOfReshapeParam> shapeParams = {
        std::make_tuple(InputShape{ 2, 3, 128, 256 }, ShapeDescriptor{ 0,  0, 64, 512 }, true),
        std::make_tuple(InputShape{ 2, 3, 128, 256 }, ShapeDescriptor{ 3,  2, 64, 512 }, false),
        std::make_tuple(InputShape{ 2, 3,   0, 256 }, ShapeDescriptor{ 3,  8,  0, 512 }, false),
        std::make_tuple(InputShape{ 2, 3, 128, 256 }, ShapeDescriptor{ 2,  3, -1,  64 }, false),
        std::make_tuple(InputShape{ 2, 3, 128, 256 }, ShapeDescriptor{ 2, -1,  0      }, true),
        std::make_tuple(InputShape{ 2, 5,   5,  24 }, ShapeDescriptor{ 0, -1,  4      }, true),
        std::make_tuple(InputShape{ 2, 5,   5,   0 }, ShapeDescriptor{ 0,  4          }, false),
};

INSTANTIATE_TEST_SUITE_P(smoke_accuracy, OutShapeOfReshapeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapeParams),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        OutShapeOfReshapeLayerTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
