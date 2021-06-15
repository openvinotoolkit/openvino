// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [fft_kernel:implementation]
#include "fft_kernel.hpp"

#include <ie_layouts.h>

#include <opencv2/opencv.hpp>

#include "fft_op.hpp"

using namespace TemplateExtension;

FFTImpl::FFTImpl(const std::shared_ptr<ngraph::Node>& node) {
    auto castedNode = std::dynamic_pointer_cast<FFTOp>(node);
    if (!castedNode)
        IE_THROW() << "Cannot create implementation for unknown operation!";
    if (castedNode->inputs().size() != 1 || castedNode->outputs().size() != 1)
        IE_THROW() << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
    if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
        IE_THROW() << "Cannot create implementation for op with dynamic shapes!";
    if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
        IE_THROW() << "Operation supports only FP32 tensors.";
    inpShape = castedNode->get_input_shape(0);
    outShape = castedNode->get_output_shape(0);
    inverse = castedNode->inverse;
}

InferenceEngine::StatusCode FFTImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc* resp) noexcept {
    std::vector<InferenceEngine::DataConfig> inDataConfig;
    std::vector<InferenceEngine::DataConfig> outDataConfig;
    InferenceEngine::SizeVector order(inpShape.size());
    std::iota(order.begin(), order.end(), 0);

    // Allow any offset before data
    size_t offset((std::numeric_limits<size_t>::max)());

    // Input shape
    InferenceEngine::DataConfig inpConf;
    inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, inpShape, {inpShape, order, offset});
    inDataConfig.push_back(inpConf);

    // Output shape
    InferenceEngine::DataConfig outConf;
    outConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outShape, {outShape, order, offset});
    outDataConfig.push_back(outConf);

    InferenceEngine::LayerConfig layerConfig;
    layerConfig.inConfs = inDataConfig;
    layerConfig.outConfs = outDataConfig;

    conf.push_back(layerConfig);
    return InferenceEngine::StatusCode::OK;
}

InferenceEngine::StatusCode FFTImpl::init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        if (config.inConfs.size() != 1 || config.outConfs.size() != 1) {
            IE_THROW() << "Operation cannot be initialized with incorrect number of inputs/outputs!";
        }

        if (config.outConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32 ||
            config.inConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32) {
            IE_THROW() << "Operation supports only FP32 precisions!";
        }
    } catch (InferenceEngine::Exception& ex) {
        if (resp) {
            strncpy(resp->msg, error.c_str(), sizeof(resp->msg) - 1);
            resp->msg[sizeof(resp->msg) - 1] = 0;
        }
        return InferenceEngine::GENERAL_ERROR;
    }
    return InferenceEngine::OK;
}

static cv::Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob) {
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->getTensorDesc().getDims();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob->getTensorDesc().getPrecision();
    CV_Assert(precision == InferenceEngine::Precision::FP32);
    return cv::Mat(size, CV_32F, (void*)blob->buffer());
}

InferenceEngine::StatusCode FFTImpl::execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                             InferenceEngine::ResponseDesc* resp) noexcept {
    cv::Mat inp = infEngineBlobToMat(inputs[0]);
    cv::Mat out = infEngineBlobToMat(outputs[0]);

    const int n = inp.size[0];
    const int h = inp.size[2];
    const int w = inp.size[3];
    cv::Mat complex(h, w, CV_32FC2), interleavedOut(h, w, CV_32FC2);
    for (int i = 0; i < n; ++i) {
        std::vector<cv::Mat> components = {cv::Mat(h, w, CV_32F, inp.ptr<float>(i, 0)), cv::Mat(h, w, CV_32F, inp.ptr<float>(i, 1))};
        cv::merge(components, complex);

        if (!inverse)
            cv::dft(complex, interleavedOut);
        else
            cv::idft(complex, interleavedOut, cv::DFT_SCALE);

        components = {cv::Mat(h, w, CV_32F, out.ptr<float>(i, 0)), cv::Mat(h, w, CV_32F, out.ptr<float>(i, 1))};
        cv::split(interleavedOut, components);
    }
    return InferenceEngine::OK;
}
//! [fft_kernel:implementation]
