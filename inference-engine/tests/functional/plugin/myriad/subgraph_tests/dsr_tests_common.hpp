// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/myriad_common_test_utils.hpp"

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>

#include "vpu/private_plugin_config.hpp"

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>

namespace LayerTestsUtils {
namespace vpu {

using DataType = ngraph::element::Type;
using DataShape = ngraph::Shape;

struct DataShapeWithUpperBound {
    DataShape shape;
    DataShape upperBoundShape;
};

class DSR_TestsCommon : public LayerTestsUtils::LayerTestsCommon {
protected:
    std::unordered_map<std::string, DataShape> m_shapes;
    ngraph::ParameterVector m_parameterVector;

    virtual std::shared_ptr<ngraph::Node> createInputSubgraphWithDSR(
            const DataType& inDataType, const DataShapeWithUpperBound& shapes,
            const std::string& suffix = "") {
        const auto inDataParam = std::make_shared<ngraph::opset3::Parameter>(
                inDataType, shapes.upperBoundShape);
        inDataParam->set_friendly_name(inDataParam->get_friendly_name() + suffix);
        const auto inDataShapeParam = std::make_shared<ngraph::opset3::Parameter>(
                ngraph::element::i32, ngraph::Shape{shapes.shape.size()});
        inDataShapeParam->set_friendly_name(inDataParam->get_friendly_name() + "/shape");

        m_shapes[inDataShapeParam->get_friendly_name()] = shapes.shape;
        m_parameterVector.push_back(inDataParam);
        m_parameterVector.push_back(inDataShapeParam);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
                inDataParam, inDataShapeParam);

        return dsr;
    }

    virtual std::shared_ptr<ngraph::Node> createTestedOp() = 0;

    void SetUp() override {
        SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
        configuration[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
        if (CommonTestUtils::vpu::CheckMyriad2()) {
            configuration[VPU_CONFIG_KEY(DISABLE_REORDER)] = CONFIG_VALUE(YES);
        }

        const auto testedOp = createTestedOp();
        const auto result = std::make_shared<ngraph::opset3::Result>(testedOp);

        function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{result},
                m_parameterVector,
                "DSR-" + std::string(testedOp->get_type_name()));
        testedOp->set_output_type(0, testedOp->get_input_element_type(0), ngraph::PartialShape::dynamic(
                testedOp->get_output_partial_shape(0).rank()));
        ::vpu::DynamicToStaticShape().transform(function);
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

    void Validate() override {
        for (const auto& op : function->get_ordered_ops()) {
            if (const auto dsr = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(op)) {
                dsr->setMode(ngraph::vpu::op::DynamicShapeResolverMode::INFER_DYNAMIC_SHAPE);
            }
        }
        function->validate_nodes_and_infer_types();

        LayerTestsCommon::Validate();

        for (const auto& op : function->get_ordered_ops()) {
            if (const auto dsr = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(op)) {
                dsr->setMode(ngraph::vpu::op::DynamicShapeResolverMode::INFER_UPPER_BOUND_SHAPE);
            }
        }
    }
};

}  // namespace vpu
}  // namespace LayerTestsUtils
