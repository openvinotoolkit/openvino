// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include <legacy/ngraph_ops/crop_ie.hpp>

#include "ie_core.hpp"
#include "single_layer_tests/crop.hpp"

namespace LayerTestsDefinitions {

    std::string Crop4DLayerTest::getTestCaseName(const testing::TestParamInfo<crop4DParamsTuple> &obj) {
        cropParams currentCropParams;
        InferenceEngine::SizeVector inputShape;
        std::vector<int64_t> axes;
        std::vector<int64_t> dim;
        std::vector<int64_t> offset;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> config;
        std::tie(currentCropParams, netPrecision, targetName, config) = obj.param;
        std::tie(inputShape, axes, dim, offset) = currentCropParams;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "axes=" << CommonTestUtils::vec2str(axes) << "_";
        result << "dim=" << CommonTestUtils::vec2str(dim) << "_";
        result << "offset=" << CommonTestUtils::vec2str(offset) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetName << "_";
        for (auto const& configItem : config) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

    void Crop4DLayerTest::SetUp() {
        cropParams currentCropParams;
        InferenceEngine::SizeVector inputShape;
        std::vector<int64_t> axes;
        std::vector<int64_t> dim;
        std::vector<int64_t> offset;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(currentCropParams, netPrecision, targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());

        std::tie(inputShape, axes, dim, offset) = currentCropParams;
        auto totalSize = std::accumulate(inputShape.begin(), inputShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto input = params[0];

        auto crop = std::make_shared <ngraph::op::CropIE>(input, axes, dim, offset);
        //Without activation the model is considered trivial for GNA
        auto activation = std::make_shared<ngraph::op::Relu>(crop);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(activation)};
        function = std::make_shared<ngraph::Function>(results, params, "CropTest");
    }

    InferenceEngine::Blob::Ptr Crop4DLayerTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();
        auto precision = info.getPrecision();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        for (size_t i = 0; i < blob->size(); i++) {
            float value = i % 16;
            if (typeid(precision) == typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
                rawBlobDataPtr[i] = ngraph::float16(value).to_bits();
            } else {
                rawBlobDataPtr[i] = value;
            }
        }
        return blob;
    }

    void Crop4DLayerTest::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        LoadNetwork();
        Infer();

        function = GenerateNgraphFriendlyModel();
        Validate();
    }

    std::shared_ptr<ngraph::Function> Crop4DLayerTest::GenerateNgraphFriendlyModel() {
        InferenceEngine::Precision netPrecision = std::get<1>(this->GetParam());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        cropParams currentCropParams;
        InferenceEngine::SizeVector inputShape;
        std::vector<int64_t> axes;
        std::vector<int64_t> dim;
        std::vector<int64_t> offset;
        std::vector<int64_t> stride;
        currentCropParams = std::get<0>(this->GetParam());
        std::tie(inputShape, axes, dim, offset) = currentCropParams;

        auto params = ngraph::builder::makeParams(ngPrc, { inputShape });
        auto input = params[0];
        auto sliceBegin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ inputShape.size() }, offset);
        std::transform(dim.begin(), dim.end(), offset.begin(), dim.begin(), std::plus<int64_t>());
        auto sliceEnd = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ inputShape.size() }, dim);

        std::vector<int64_t> beginMask = { };
        std::vector<int64_t> endMask = { };
        int64_t currentOffset = 0;
        for (int64_t o : offset) {
            currentOffset = (o == 0) ? 1 : 0;
            beginMask.push_back(currentOffset);
            endMask.push_back(currentOffset);
            stride.push_back(1);
        }
        auto sliceStride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ inputShape.size() }, stride);
        auto sslice = std::make_shared<ngraph::opset1::StridedSlice>(input, sliceBegin, sliceEnd, sliceStride,
                                                                     beginMask, endMask);
        auto activation = std::make_shared<ngraph::op::Relu>(sslice);

        ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(activation) };
        return std::make_shared<ngraph::Function>(results, params, "CropTest");
    }


    TEST_P(Crop4DLayerTest, CompareWithRefs) {
        Run();
    };
}  // namespace LayerTestsDefinitions
