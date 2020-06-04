// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>

#include "vpu/private_plugin_config.hpp"

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>

namespace {

using DataType = ngraph::element::Type;
using DataShape = ngraph::Shape;

struct FunctionWithTestedOp {
    std::shared_ptr<ngraph::Function> function;
    std::shared_ptr<ngraph::Node> testedOp;
};

class DSR_TestsCommon : public LayerTestsUtils::LayerTestsCommon {
protected:
    std::unordered_map<std::string, DataShape> m_shapes;
    ngraph::ParameterVector m_parameterVector;

    virtual std::shared_ptr<ngraph::Node> createInputSubgraphWithDSR(
            const DataType& inDataType, const DataShape& inDataShape) {
        const auto inDataParam = std::make_shared<ngraph::opset3::Parameter>(
                inDataType, inDataShape);
        const auto inDataShapeParam = std::make_shared<ngraph::opset3::Parameter>(
                ngraph::element::i32, ngraph::Shape{inDataShape.size()});
        inDataShapeParam->set_friendly_name(inDataParam->get_friendly_name() + "/shape");

        m_shapes[inDataShapeParam->get_friendly_name()] = inDataShape;
        m_parameterVector.push_back(inDataParam);
        m_parameterVector.push_back(inDataShapeParam);

        const auto dataConstFactor = std::make_shared<ngraph::opset3::Constant>(
                inDataType, ngraph::Shape{}, 1.5f);
        const auto shapeConstFactor = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i32, ngraph::Shape{inDataShape.size()}, 1);
        const auto inDataCopy = std::make_shared<ngraph::opset3::Multiply>(
                inDataParam, dataConstFactor);
        const auto inDataShapeCopy = std::make_shared<ngraph::opset3::Multiply>(
                inDataShapeParam, shapeConstFactor);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
                inDataCopy, inDataShapeCopy);

        return dsr;
    }

    virtual FunctionWithTestedOp createFunctionWithTestedOp() = 0;

    void SetUp() override {
        SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
        configuration[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
        configuration[VPU_CONFIG_KEY(DISABLE_REORDER)] = CONFIG_VALUE(YES);

        const auto& infFunctionWithTestedOp = createFunctionWithTestedOp();
        function = infFunctionWithTestedOp.function;
        const auto testedOp = infFunctionWithTestedOp.testedOp;
        testedOp->set_output_type(0, testedOp->get_input_element_type(0), ngraph::PartialShape::dynamic(
                testedOp->get_output_partial_shape(0).rank()));
        vpu::DynamicToStaticShape().transform(function);
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        const auto& shapeIt = m_shapes.find(info.name());
        if (shapeIt == m_shapes.end()) {
            return LayerTestsCommon::GenerateInput(info);
        }

        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
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
        function->validate_nodes_and_infer_types();
    }
};

}  // namespace
