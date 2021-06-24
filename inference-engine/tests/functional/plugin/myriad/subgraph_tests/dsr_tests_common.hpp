// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/myriad_common_test_utils.hpp"

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>

#include "vpu/private_plugin_config.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>

namespace LayerTestsUtils {
namespace vpu {

using DataType = ngraph::element::Type;
using DataShape = ngraph::Shape;

struct DataShapeWithUpperBound {
    DataShape shape;
    DataShape upperBoundShape;
};

class DSR_TestsCommon : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    std::unordered_map<std::string, DataShape> m_shapes;
    ngraph::ParameterVector m_parameterVector;
    ngraph::ResultVector m_additionalResults;

    std::shared_ptr<ngraph::opset3::Parameter> createParameter(
            const ngraph::element::Type& element_type,
            const ngraph::PartialShape& shape) {
        m_parameterVector.push_back(std::make_shared<ngraph::op::Parameter>(element_type, shape));
        return m_parameterVector.back();
    }

    virtual std::shared_ptr<ngraph::Node> createInputSubgraphWithDSR(
            const DataType& inDataType, const DataShapeWithUpperBound& shapes) {
        const auto inDataParam = std::make_shared<ngraph::opset3::Parameter>(
                inDataType, shapes.upperBoundShape);
        const auto inDataShapeParam = std::make_shared<ngraph::opset3::Parameter>(
                ngraph::element::i32, ngraph::Shape{shapes.shape.size()});
        inDataShapeParam->set_friendly_name(inDataParam->get_friendly_name() + "/shape");

        m_shapes[inDataShapeParam->get_friendly_name()] = shapes.shape;
        m_parameterVector.push_back(inDataParam);
        m_parameterVector.push_back(inDataShapeParam);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
                inDataParam, inDataShapeParam, ngraph::vpu::op::DynamicShapeResolverMode::INFER_DYNAMIC_SHAPE);

        return dsr;
    }

    virtual std::shared_ptr<ngraph::Node> createTestedOp() = 0;

    void SetUp() override {
        // TODO: Replace CONSTANT_FOLDING to INTERPRETER
        SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
        if (CommonTestUtils::vpu::CheckMyriad2()) {
            configuration[InferenceEngine::MYRIAD_DISABLE_REORDER] = CONFIG_VALUE(YES);
        }

        const auto testedOp = createTestedOp();
        ngraph::ResultVector results{};
        for (const auto& output : testedOp->outputs()) {
            results.emplace_back(std::make_shared<ngraph::opset3::Result>(output));
        }
        results.insert(results.end(), m_additionalResults.begin(), m_additionalResults.end());

        function = std::make_shared<ngraph::Function>(
                results,
                m_parameterVector,
                "DSR-" + std::string(testedOp->get_type_name()));
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        const auto& shapeIt = m_shapes.find(info.name());
        if (shapeIt == m_shapes.end()) {
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 5, 1, 1);
        }

        auto blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto dataPtr = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rwmap().as<int32_t*>();
        for (size_t i = 0; i < blob->size(); ++i) {
            dataPtr[i] = shapeIt->second[i];
        }

        return blob;
    }
};

}  // namespace vpu
}  // namespace LayerTestsUtils
