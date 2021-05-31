// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [fft_kernel:implementation]
#include "fft_kernel.hpp"

#include <details/ie_so_loader.h>
#include <ie_layouts.h>
#include <opencv2/core/core_c.h>

#include "fft_op.hpp"

InferenceEngine::details::SharedObjectLoader so;
using cvCreateMatHeaderF = CvMat*(int, int, int);
using cvCreateMatF = CvMat*(int, int, int);
using cvSetDataF = void(CvArr*, void*, int);
using cvReleaseMatF = void(CvMat**);
using cvMergeF = void(const CvArr*, const CvArr*, const CvArr*, const CvArr*, CvArr*);
using cvSplitF = void(const CvArr*, CvArr*, CvArr*, CvArr*, CvArr*);
using cvDftF = void(const CvArr*, CvArr*, int, int);

bool loadOpenCV() {
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        try {
#ifdef _WIN32
            so = InferenceEngine::details::SharedObjectLoader("opencv_core.dll");
#elif defined(__APPLE__)
            so = InferenceEngine::details::SharedObjectLoader("libopencv_core.dylib");
#else
            so = InferenceEngine::details::SharedObjectLoader("libopencv_core.so");
#endif
        } catch (InferenceEngine::Exception&) {
            return false;
        }
    }
    return loaded;
}

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
        if (!loadOpenCV()) {
            IE_THROW() << "Failed to load OpenCV!";
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

InferenceEngine::StatusCode FFTImpl::execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                             InferenceEngine::ResponseDesc* resp) noexcept {
    static auto cvSetData = reinterpret_cast<cvSetDataF*>(so.get_symbol("cvSetData"));
    static auto cvCreateMatHeader = reinterpret_cast<cvCreateMatHeaderF*>(so.get_symbol("cvCreateMatHeader"));
    static auto cvCreateMat = reinterpret_cast<cvCreateMatF*>(so.get_symbol("cvCreateMat"));
    static auto cvMerge = reinterpret_cast<cvMergeF*>(so.get_symbol("cvMerge"));
    static auto cvSplit = reinterpret_cast<cvSplitF*>(so.get_symbol("cvSplit"));
    static auto cvDFT = reinterpret_cast<cvDftF*>(so.get_symbol("cvDFT"));
    static auto cvReleaseMat = reinterpret_cast<cvReleaseMatF*>(so.get_symbol("cvReleaseMat"));

    float* inpData = inputs[0]->buffer();
    float* outData = outputs[0]->buffer();

    std::vector<size_t> dims = inputs[0]->getTensorDesc().getDims();
    const size_t n = dims[0];
    const size_t h = dims[2];
    const size_t w = dims[3];
    const size_t planeSize = h * w * 2;
    CvMat* real = cvCreateMatHeader(h, w, CV_32FC1);
    CvMat* imag = cvCreateMatHeader(h, w, CV_32FC1);
    CvMat* complex = cvCreateMat(h, w, CV_32FC2);
    CvMat* interleavedOut = cvCreateMat(h, w, CV_32FC2);
    for (size_t i = 0; i < n; ++i) {
        cvSetData(real, reinterpret_cast<void*>(inpData + i * planeSize), w * sizeof(float));
        cvSetData(imag, reinterpret_cast<void*>(inpData + i * planeSize + h * w), w * sizeof(float));
        cvMerge(real, imag, nullptr, nullptr, complex);

        if (!inverse)
            cvDFT(complex, interleavedOut, CV_DXT_FORWARD, 0);
        else
            cvDFT(complex, interleavedOut, CV_DXT_INVERSE | CV_DXT_SCALE, 0);

        cvSetData(real, reinterpret_cast<void*>(outData + i * planeSize), w * sizeof(float));
        cvSetData(imag, reinterpret_cast<void*>(outData + i * planeSize + h * w), w * sizeof(float));
        cvSplit(interleavedOut, real, imag, nullptr, nullptr);
    }
    cvReleaseMat(&real);
    cvReleaseMat(&imag);
    cvReleaseMat(&complex);
    cvReleaseMat(&interleavedOut);
    return InferenceEngine::OK;
}
//! [fft_kernel:implementation]
