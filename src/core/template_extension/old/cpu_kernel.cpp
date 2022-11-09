// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_kernel.hpp"

#include <ie_layouts.h>

#include "op.hpp"

using namespace TemplateExtension;

//! [cpu_implementation:ctor]
OpImplementation::OpImplementation(const std::shared_ptr<ngraph::Node>& node) {
    try {
        auto castedNode = std::dynamic_pointer_cast<Operation>(node);
        if (!castedNode)
            IE_THROW() << "Cannot create implementation for unknown operation!";
        if (castedNode->inputs().size() != 1 || castedNode->outputs().size() != 1)
            IE_THROW() << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
        if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
            IE_THROW() << "Cannot create implementation for op with dynamic shapes!";
        if (castedNode->get_input_shape(0).size() != 4 || castedNode->get_output_shape(0).size() != 4)
            IE_THROW() << "Operation supports only 4d tensors for input and output.";
        if (castedNode->get_input_element_type(0) != ngraph::element::f32 ||
            castedNode->get_output_element_type(0) != ngraph::element::f32)
            IE_THROW() << "Operation supports only FP32 tensors.";
        add = castedNode->getAddAttr();
        inShape = castedNode->get_input_shape(0);
        outShape = castedNode->get_output_shape(0);
    } catch (InferenceEngine::Exception& ex) {
        error = ex.what();
    }
}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode OpImplementation::getSupportedConfigurations(
    std::vector<InferenceEngine::LayerConfig>& conf,
    InferenceEngine::ResponseDesc* resp) noexcept {
    auto createConfig = [](const InferenceEngine::SizeVector inShape,
                           const InferenceEngine::SizeVector& outShape,
                           bool planar) {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = false;
        InferenceEngine::DataConfig inData;
        InferenceEngine::DataConfig outData;
        InferenceEngine::SizeVector order = {0, 1, 2, 3};
        // Allow any offset before data
        size_t offset((std::numeric_limits<size_t>::max)());
        if (planar) {
            inData.desc =
                InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inShape, {inShape, order, offset});
            config.inConfs.push_back(inData);
            outData.desc =
                InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outShape, {outShape, order, offset});
            config.outConfs.push_back(outData);
        } else {
            // Add blocked (nChw8c) format
            auto div_up = [](const int a, const int b) -> int {
                if (!b)
                    return 0;
                return (a + b - 1) / b;
            };

            order.push_back(1);
            InferenceEngine::SizeVector inBlkDims = inShape;
            inBlkDims[1] = div_up(static_cast<int>(inBlkDims[1]), 8);
            inBlkDims.push_back(8);
            InferenceEngine::SizeVector outBlkDims = outShape;
            outBlkDims[1] = div_up(static_cast<int>(outBlkDims[1]), 8);
            outBlkDims.push_back(8);
            inData.desc =
                InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inShape, {inBlkDims, order, offset});
            config.inConfs.push_back(inData);
            outData.desc =
                InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outShape, {outBlkDims, order, offset});
            config.outConfs.push_back(outData);
        }
        return config;
    };
    if (!error.empty()) {
        if (resp) {
            strncpy(resp->msg, error.c_str(), sizeof(resp->msg) - 1);
            resp->msg[sizeof(resp->msg) - 1] = 0;
        }
        return InferenceEngine::GENERAL_ERROR;
    }
    // Add planar format
    conf.emplace_back(createConfig(inShape, outShape, true));
    // Add blocked format nChw8c
    conf.emplace_back(createConfig(inShape, outShape, false));
    return InferenceEngine::OK;
}
//! [cpu_implementation:getSupportedConfigurations]

//! [cpu_implementation:init]
InferenceEngine::StatusCode OpImplementation::init(InferenceEngine::LayerConfig& config,
                                                   InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        if (config.inConfs.size() != 1 || config.outConfs.size() != 1) {
            IE_THROW() << "Operation cannot be initialized with incorrect number of inputs/outputs!";
        }

        if (config.inConfs[0].desc.getDims().size() != 4 || config.outConfs[0].desc.getDims().size() != 4) {
            IE_THROW() << "Operation can be initialized only with 4d input/output tensors!";
        }

        if (config.outConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32 ||
            config.inConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32) {
            IE_THROW() << "Operation supports only FP32 precisions!";
        }
    } catch (InferenceEngine::Exception& ex) {
        error = ex.what();
        if (resp) {
            strncpy(resp->msg, error.c_str(), sizeof(resp->msg) - 1);
            resp->msg[sizeof(resp->msg) - 1] = 0;
        }
        return InferenceEngine::GENERAL_ERROR;
    }

    return InferenceEngine::OK;
}
//! [cpu_implementation:init]

//! [cpu_implementation:execute]
InferenceEngine::StatusCode OpImplementation::execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                                      std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                                      InferenceEngine::ResponseDesc* resp) noexcept {
    const float* src_data =
        inputs[0]->cbuffer().as<const float*>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    float* dst_data =
        outputs[0]->buffer().as<float*>() + outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

    for (size_t i = 0; i < inputs[0]->size(); i++) {
        dst_data[i] = src_data[i] + add;
    }
    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]
