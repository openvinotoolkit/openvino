// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef USE_OPENCV
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <ie_compound_blob.h>
#include <precision_utils.h>
#include <ie_precision.hpp>
#include <ie_plugin_config.hpp>

#include "tests_common.hpp"
#include "format_reader_ptr.h"
#include "single_layer_common.hpp"

#include "functional_test_utils/plugin_cache.hpp"

#include "ie_preprocess_data.hpp"

#include <map>
#include <functional_test_utils/precision_utils.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph_functions/builders.hpp>
#include <functional_test_utils/blob_utils.hpp>

using namespace ::testing;
using namespace InferenceEngine;

template <Precision::ePrecision PRC>
Blob::Ptr img2Blob(const std::vector<cv::Mat>& imgs, Layout layout) {
    using data_t = typename PrecisionTrait<PRC>::value_type;

    if (imgs.empty()) {
        IE_THROW() << "No images to create blob from";
    }

    // get image value in correct format
    static const auto img_value = [] (const cv::Mat& img, size_t h, size_t w, size_t c) -> data_t {
        switch (img.type())
        {
            case CV_8UC1: return img.at<uchar>(h, w);
            case CV_8UC2: return img.at<cv::Vec2b>(h, w)[c];
            case CV_8UC3: return img.at<cv::Vec3b>(h, w)[c];
            case CV_8UC4: return img.at<cv::Vec4b>(h, w)[c];
            case CV_32FC3: return img.at<cv::Vec3f>(h, w)[c];
            case CV_32FC4: return img.at<cv::Vec4f>(h, w)[c];
            default:
                IE_THROW() << "Image type is not recognized";
        }
    };

    size_t channels = imgs[0].channels();
    size_t height = imgs[0].size().height;
    size_t width = imgs[0].size().width;

    SizeVector dims = {imgs.size(), channels, height, width};
    Blob::Ptr resultBlob = make_shared_blob<data_t>(TensorDesc(PRC, dims, layout));
    resultBlob->allocate();

    data_t* blobData = resultBlob->buffer().as<data_t*>();

    for (size_t i = 0; i < imgs.size(); ++i) {
        auto& img = imgs[i];
        auto batch_offset = i * channels * height * width;

        switch (layout) {
            case Layout::NCHW: {
                for (size_t c = 0; c < channels; c++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            blobData[batch_offset + c * width * height + h * width + w] =
                                img_value(img, h, w, c);
                        }
                    }
                }
            }
            break;
            case Layout::NHWC: {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        for (size_t c = 0; c < channels; c++) {
                            blobData[batch_offset + h * width * channels + w * channels + c] =
                                img_value(img, h, w, c);
                        }
                    }
                }
            }
            break;
            default:
                IE_THROW() << "Inconsistent input layout for image processing: " << layout;
        }
    }
    return resultBlob;
}

template <Precision::ePrecision PRC>
Blob::Ptr img2Blob(cv::Mat &img, Layout layout) {
    return img2Blob<PRC>(std::vector<cv::Mat>({img}), layout);
}

// base class with common functionality for test fixtures
template<typename Params>
class Base : public TestsCommon, public WithParamInterface<Params> {
protected:
    std::string _device;
    Precision _netPrc = Precision(Precision::UNSPECIFIED);
    TBlob<uint8_t>::Ptr _weights;
    SizeVector _netDims;
    Precision _inputPrecision = Precision(Precision::UNSPECIFIED);
    float _threshold = 0.f;
    Layout _inputLayout = Layout::ANY;
    ResizeAlgorithm _resAlg = ResizeAlgorithm::NO_RESIZE;
    ColorFormat _colorFormat = ColorFormat::RAW;
    ROI _cropRoi = {};
    bool _isAsync = false;
    constexpr static const int _maxRepeat = 2;

    std::map<std::string, std::string> device_config;

    std::shared_ptr<InferenceEngine::Core> ie;

    void SetUp() override {
        TestsCommon::SetUp();

        ie = PluginCache::get().ie();
    }

    void TearDown() override {
    }

public:
    std::shared_ptr<ngraph::Function> createSubgraph(const SizeVector &dims, InferenceEngine::Precision prc = InferenceEngine::Precision::FP32) {
        ngraph::element::Type type = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prc);

        auto param0 = std::make_shared<ngraph::op::Parameter>(type, ngraph::PartialShape{dims});
        auto relu1 = std::make_shared<ngraph::opset1::Relu>(param0);

        ngraph::ParameterVector params = {param0};
        ngraph::ResultVector results = {std::make_shared<ngraph::op::Result>(relu1)};

        auto fn_ptr = std::make_shared<ngraph::Function>(results, params);
        return fn_ptr;
    }

    cv::ColorConversionCodes toCvtColorType(ColorFormat fmt) {
        // Note: OpenCV matrices are always in BGR format by default
        switch (fmt) {
            case ColorFormat::BGRX: return cv::COLOR_BGR2BGRA;
            case ColorFormat::RGBX: return cv::COLOR_BGR2RGBA;
            case ColorFormat::RGB: return cv::COLOR_BGR2RGB;
            default: IE_THROW() << "Color format " << fmt << " not found";
        }
        return cv::COLOR_COLORCVT_MAX;
    }

    void auxDownscale(cv::Mat& img, cv::InterpolationFlags interpolation, int bonus_divider = 1) {
        cv::resize(img, img, cv::Size(_netDims[3] / 2, _netDims[2] / 2) / bonus_divider, 0, 0,
            interpolation);
    }

    InferenceEngine::ROI auxDownscaledRoi() {
        const auto make_even = [] (size_t v) { return v % 2 != 0 ? v + 1 : v; };
        auto h = make_even(_netDims[3]), w = make_even(_netDims[2]);
        if (h % 4 != 0 || w % 4 != 0) {
            return {0, 0, 0, h, w};
        } else {
            return {0, 0, 0, h / 2, w / 2};
        }
    }
};

// base resize parameters used by test fixtures
template<typename ImageParam, typename LayoutParam = Layout>
using resize_params = std::tuple<
    std::string,                          // Plugin name
    std::tuple<
            Precision,                    // Network precision
            SizeVector,                   // Net input sizes
            std::pair<Precision, float>,  // Input data precision and threshold
            LayoutParam,                  // Input data layout
            ResizeAlgorithm,              // Resize algorithm kind
            ColorFormat,                  // Input color format kind
            ROI,                          // Cropped ROI coordinates
            bool                          // Infer modes: true = Async, false = Sync
    >
>;

namespace {
// division operator for ROI
InferenceEngine::ROI operator/(const InferenceEngine::ROI& roi, size_t divider) {
    return InferenceEngine::ROI{
        roi.id,
        roi.posX / divider,
        roi.posY / divider,
        roi.sizeX / divider,
        roi.sizeY / divider
    };
}
}  // anonymous namespace

InferenceEngine::ROI getRandomROI(const cv::Size& picSize)
{
    ROI rect;

    rect.posX =  static_cast <size_t>((picSize.width*0.75) * ((double)std::rand()/(double)RAND_MAX));
    rect.posY =  static_cast <size_t>((picSize.height*0.75) * ((double)std::rand()/(double)RAND_MAX));
    rect.sizeX = static_cast <size_t>((picSize.width/4) * ((double)std::rand()/(double)RAND_MAX));
    rect.sizeY = static_cast <size_t>((picSize.height/4) * ((double)std::rand()/(double)RAND_MAX));

    // According to initScratchLinear function picture width should be >= 2
    if (rect.sizeX < 2)
        rect.sizeX = picSize.width/5; //20% of picture width to fit exactly in the last 25% of the picture.

    if (rect.sizeY < 2)
        rect.sizeY = rect.sizeX;

    if (rect.posX + rect.sizeX > picSize.width)
        rect.sizeX = picSize.width - rect.posX;

    if (rect.posY + rect.sizeY > picSize.height)
        rect.sizeY = picSize.height - rect.posY;

    return rect;
}

class RandomROITest: public Base<resize_params<std::string, Layout>>
{
protected:
    void SetUp() override {
        Base<resize_params<std::string, Layout>>::SetUp();
        _device = std::get<0>(GetParam());
        std::pair<Precision, float> _inPrcThresh;
        std::tie(
                _netPrc,
                _netDims,
                _inPrcThresh,
                _inputLayout,
                _resAlg,
                _colorFormat,
                _cropRoi,
                _isAsync
        ) = std::get<1>(GetParam());

        if (((_colorFormat == BGRX) || (_colorFormat == RGBX)) && (_inputLayout != NHWC))
        {
            IE_THROW() << "The color format with the layout aren't compatible.";
        }

        _inputPrecision = _inPrcThresh.first;
        _threshold = _inPrcThresh.second;

        if (_device == "HETERO")
            device_config["TARGET_FALLBACK"] = "GPU,CPU";
    }
};

TEST_P(RandomROITest, PreprocRandomROITest)
{
    auto fn_ptr = createSubgraph(_netDims);
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setPrecision(_inputPrecision);
    net.getInputsInfo().begin()->second->setLayout(_inputLayout);
    net.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(_resAlg);
    net.getInputsInfo().begin()->second->getPreProcess().setColorFormat(_colorFormat);

    auto execNet = ie->LoadNetwork(net, _device, device_config);
    auto req = execNet.CreateInferRequest();

    Blob::Ptr blob;
    Blob::Ptr yBlob;
    Blob::Ptr uvBlob;

    auto w = 200;
    auto h = 200;
    switch (_colorFormat)
    {
        case BGR:
            switch (_inputPrecision)
            {
                case Precision::U8:
                {
                    cv::Mat mat(h, w, CV_8UC3);
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    blob = img2Blob<Precision::U8>(mat, _inputLayout);
                    break;
                }
                case Precision::FP16:
                {
                    cv::Mat mat(h, w, CV_16FC3,  cv::Scalar(0,0, 255));
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    blob = img2Blob<Precision::FP16>(mat, _inputLayout);
                    break;
                }
                case Precision::FP32:
                {
                    cv::Mat mat(h, w, CV_32FC3);
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    blob = img2Blob<Precision::FP32>(mat, _inputLayout);
                    break;
                }
                default:
                    break;
            }
            break;
        case RGB:
            switch (_inputPrecision)
            {
                case Precision::U8:
                {
                    cv::Mat mat(h, w, CV_8UC3);
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    cv::cvtColor(mat, mat, toCvtColorType(_colorFormat));
                    blob = img2Blob<Precision::U8>(mat, _inputLayout);
                    break;
                }
                case Precision::FP16:
                {
                    cv::Mat mat(h, w, CV_16FC3);
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    cv::cvtColor(mat, mat, toCvtColorType(_colorFormat));
                    blob = img2Blob<Precision::FP16>(mat, _inputLayout);
                    break;
                }
                case Precision::FP32:
                {
                    cv::Mat mat(h, w, CV_32FC3);
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    cv::cvtColor(mat, mat, toCvtColorType(_colorFormat));
                    blob = img2Blob<Precision::FP32>(mat, _inputLayout);
                    break;
                }
                default:
                    break;
            }

            break;
        case BGRX:
        case RGBX:
            switch (_inputPrecision)
            {
                case Precision::U8:
                {
                    cv::Mat mat(h, w, CV_8UC4);
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    cv::cvtColor(mat, mat, toCvtColorType(_colorFormat));
                    blob = img2Blob<Precision::U8>(mat, _inputLayout);
                    break;
                }
                case Precision::FP16:
                {
                    cv::Mat mat(h, w, CV_16FC4);
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    cv::cvtColor(mat, mat, toCvtColorType(_colorFormat));
                    blob = img2Blob<Precision::FP16>(mat, _inputLayout);
                    break;
                }
                case Precision::FP32:
                {
                    cv::Mat mat(h, w , CV_32FC4);
                    cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
                    cv::cvtColor(mat, mat, toCvtColorType(_colorFormat));
                    blob = img2Blob<Precision::FP32>(mat, _inputLayout);
                    break;
                }
                default:
                    break;
            }

            break;
        case NV12:
        {
            cv::Mat yPlane(h, w, CV_MAKE_TYPE(CV_8U, 1));
            cv::Mat uvPlane(h/2, w/2, CV_MAKE_TYPE(CV_8U, 2));

            cv::randn(yPlane, cv::Scalar::all(127), cv::Scalar::all(40.f));
            cv::randu(uvPlane, cv::Scalar::all(0), cv::Scalar::all(255));

            yBlob = img2Blob<Precision::U8>(yPlane, _inputLayout);
            uvBlob = img2Blob<Precision::U8>(uvPlane, _inputLayout);

            break;
        }
        default:
            break;
    }


    for(int i = 0; i <= 10; ++i)
    {
        ROI roi = getRandomROI(cv::Size(w, h));
        Blob::Ptr cropBlob;

        if (_colorFormat == NV12)
        {
            if (i % 2)
            {
                // New way to create NV12 ROI

                auto nv12Blob = make_shared_blob<NV12Blob>(yBlob, uvBlob);
                cropBlob = nv12Blob->createROI(roi);
            }
            else
            {
                // Old way to create NV12 ROI

                roi.sizeX += roi.sizeX % 2;
                roi.sizeY += roi.sizeY % 2;

                auto roiUV = roi/2;

                auto cropYBlob = make_shared_blob(yBlob, roi);
                auto cropUvBlob = make_shared_blob(uvBlob, roiUV);

                cropBlob = make_shared_blob<NV12Blob>(cropYBlob, cropUvBlob);
            }
        }
        else
        {
            cropBlob = make_shared_blob(blob, roi);
        }

        req.SetBlob(net.getInputsInfo().begin()->first, cropBlob);

        if (_isAsync)
        {
            req.StartAsync();
            req.Wait(InferRequest::WaitMode::RESULT_READY);
        }
        else
        {
            req.Infer();
        }
    }
}

template<typename ImageParam, typename LayoutParam = Layout>
class ResizeBase : public Base<resize_params<ImageParam, LayoutParam>> {
protected:
    bool _doColorConversion = false;
};

class CropResizeTest : public ResizeBase<std::string> {
protected:
    cv::Mat _img;

    void SetUp() override {
        ResizeBase<std::string>::SetUp();
        _device = std::get<0>(GetParam());
        std::pair<Precision, float> inPrcThresh;
        std::tie(
                _netPrc,
                _netDims,
                inPrcThresh,
                _inputLayout,
                _resAlg,
                _colorFormat,
                _cropRoi,
                _isAsync
        ) = std::get<1>(GetParam());

        _img = cv::Mat(300, 300, CV_8UC3);
        cv::randu(_img, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));

        _inputPrecision = inPrcThresh.first;
        _threshold = inPrcThresh.second;

        _doColorConversion = _colorFormat != ColorFormat::RAW && _colorFormat != ColorFormat::BGR;

        if (_device == "HETERO")
            device_config["TARGET_FALLBACK"] = "GPU,CPU";
    }

    void prepareInputandReferenceImage(Blob::Ptr& inputBlob, Blob::Ptr& refBlob) {
        // we use an image resized by openCV as a reference value.
        cv::InterpolationFlags cv_interpolation = (_resAlg == RESIZE_BILINEAR) ? cv::INTER_LINEAR : cv::INTER_AREA;

        // when no resize is performed (input size == output size), resizedImg is a shallow copy of
        // _img in case of Precision::U8 and therefore it is color converted the same way as _img!
        // doing explicit cloning prevents this
        cv::Mat resizedImg = _img.clone();

        switch (_inputPrecision) {
            case Precision::FP32: {
                cv::Mat resizedImg_;
                _img.convertTo(resizedImg_, CV_32FC3);
                cv::resize(resizedImg_, resizedImg, cv::Size(_netDims[3], _netDims[2]), 0, 0, cv_interpolation);

                if (_doColorConversion) {
                    cv::cvtColor(_img, _img, toCvtColorType(_colorFormat));
                }
                inputBlob = img2Blob<Precision::FP32>(_img, _inputLayout);
            }
                break;
            case Precision::U8: {
                cv::resize(_img, resizedImg, cv::Size(_netDims[3], _netDims[2]), 0, 0, cv_interpolation);

                if (_doColorConversion) {
                    cv::cvtColor(_img, _img, toCvtColorType(_colorFormat));
                }
                inputBlob = img2Blob<Precision::U8>(_img, _inputLayout);
            }
                break;
            default:
                IE_THROW() << "Can't resize data of inconsistent precision: " << _inputPrecision;
        }

        refBlob = img2Blob<Precision::FP32>(resizedImg, Layout::NCHW);
    }
};

TEST_P(CropResizeTest, resizeTest) {
    auto fn_ptr = createSubgraph(_netDims);
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setPrecision(_inputPrecision);
    net.getInputsInfo().begin()->second->setLayout(_inputLayout);
    net.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(_resAlg);
    net.getInputsInfo().begin()->second->getPreProcess().setColorFormat(_colorFormat);

    auto execNet = ie->LoadNetwork(net, _device, device_config);
    auto req = execNet.CreateInferRequest();

    Blob::Ptr inputBlob;
    Blob::Ptr refBlob;

    prepareInputandReferenceImage(inputBlob, refBlob);

    req.SetBlob(net.getInputsInfo().begin()->first, inputBlob);

    if (_isAsync) {
        req.StartAsync();
        req.Wait(InferRequest::WaitMode::RESULT_READY);
    } else {
        req.Infer();
    }

    Blob::Ptr outputBlob = req.GetBlob(net.getOutputsInfo().begin()->first);

    if (refBlob->size() != outputBlob->size()) {
        IE_THROW() << "reference and output blobs have different sizes!";
    }

    compare(*outputBlob, *refBlob, _threshold);
}

TEST_P(CropResizeTest, resizeAfterLoadTest) {
    auto fn_ptr = createSubgraph(_netDims);
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setPrecision(_inputPrecision);
    net.getInputsInfo().begin()->second->setLayout(_inputLayout);

    auto execNet = ie->LoadNetwork(net, _device, device_config);
    auto req = execNet.CreateInferRequest();

    Blob::Ptr inputBlob;
    Blob::Ptr refBlob;

    prepareInputandReferenceImage(inputBlob, refBlob);

    PreProcessInfo info;
    info.setResizeAlgorithm(_resAlg);
    info.setColorFormat(_colorFormat);
    req.SetBlob(net.getInputsInfo().begin()->first, inputBlob, info);

    if (_isAsync) {
        req.StartAsync();
        req.Wait(InferRequest::WaitMode::RESULT_READY);
    } else {
        req.Infer();
    }

    Blob::Ptr outputBlob = req.GetBlob(net.getOutputsInfo().begin()->first);

    if (refBlob->size() != outputBlob->size()) {
        IE_THROW() << "reference and output blobs have different sizes!";
    }

    compare(*outputBlob, *refBlob, _threshold);
}

TEST_P(CropResizeTest, cropRoiTest) {
    auto fn_ptr = createSubgraph(_netDims);
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setPrecision(_inputPrecision);
    net.getInputsInfo().begin()->second->setLayout(_inputLayout);
    net.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(_resAlg);
    net.getInputsInfo().begin()->second->getPreProcess().setColorFormat(_colorFormat);

    auto execNet = ie->LoadNetwork(net, _device, device_config);
    auto req = execNet.CreateInferRequest();

    Blob::Ptr inputBlob;
    Blob::Ptr cropRoiBlob;
    Blob::Ptr refBlob;

    // we use an image resized by openCV as a reference value.
    cv::InterpolationFlags cv_interpolation = (_resAlg == RESIZE_BILINEAR) ? cv::INTER_LINEAR : cv::INTER_AREA;

    cv::Rect location;
    location.x = _cropRoi.posX;
    location.y = _cropRoi.posY;
    location.width = _cropRoi.sizeX;
    location.height = _cropRoi.sizeY;

    auto clippedRect = location & cv::Rect(0, 0, _img.size().width, _img.size().height);
    cv::Mat imgRoi = _img(clippedRect);

    // when no resize is performed (input size == output size), resizedImg is a shallow copy of
    // _img in case of Precision::U8 and therefore it is color converted the same way as _img!
    // doing explicit cloning prevents this
    cv::Mat resizedImg = _img.clone();

    switch (_inputPrecision) {
        case Precision::FP32: {
            cv::Mat resizedImg_;
            imgRoi.convertTo(resizedImg_, CV_32FC3);
            cv::resize(resizedImg_, resizedImg, cv::Size(_netDims[3], _netDims[2]), 0, 0, cv_interpolation);

            if (_doColorConversion) {
                cv::cvtColor(_img, _img, toCvtColorType(_colorFormat));
            }

            inputBlob = img2Blob<Precision::FP32>(_img, _inputLayout);
        }
        break;
        case Precision::U8: {
            cv::resize(imgRoi, resizedImg, cv::Size(_netDims[3], _netDims[2]), 0, 0, cv_interpolation);

            if (_doColorConversion) {
                cv::cvtColor(_img, _img, toCvtColorType(_colorFormat));
            }

            inputBlob = img2Blob<Precision::U8>(_img, _inputLayout);
        }
        break;
        default:
            IE_THROW() << "Can't resize data of inconsistent precision: " << _inputPrecision;
    }
    refBlob = img2Blob<Precision::FP32>(resizedImg, Layout::NCHW);

    cropRoiBlob = make_shared_blob(inputBlob, _cropRoi);
    ASSERT_EQ(_inputPrecision, cropRoiBlob->getTensorDesc().getPrecision());

    req.SetBlob(net.getInputsInfo().begin()->first, cropRoiBlob);

    if (_isAsync) {
        req.StartAsync();
        req.Wait(InferRequest::WaitMode::RESULT_READY);
    } else {
        req.Infer();
    }

    Blob::Ptr outputBlob = req.GetBlob(net.getOutputsInfo().begin()->first);

    if (refBlob->size() != outputBlob->size()) {
        IE_THROW() << "reference and output blobs have different sizes!";
    }

    compare(*outputBlob, *refBlob, _threshold);
}

class BatchResizeTest : public ResizeBase<std::vector<std::string>> {
protected:
    std::vector<cv::Mat> _imgs;

    void SetUp() override {
        ResizeBase<std::vector<std::string>>::SetUp();

        _device = std::get<0>(GetParam());
        std::pair<Precision, float> inPrcThresh;
        std::tie(
                _netPrc,
                _netDims,
                inPrcThresh,
                _inputLayout,
                _resAlg,
                _colorFormat,
                _cropRoi,
                _isAsync
        ) = std::get<1>(GetParam());
        auto batch_size = _netDims[0];

        _imgs.reserve(batch_size);
        int h = 200, w = 200;
        for (size_t i = 0; i < batch_size; ++i) {
            cv::Mat img(h, w, CV_8UC3);
            cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
            _imgs.push_back(img);
        }

        _inputPrecision = inPrcThresh.first;
        _threshold = inPrcThresh.second;

        _doColorConversion = _colorFormat != ColorFormat::RAW && _colorFormat != ColorFormat::BGR;

        if (_device == "HETERO")
            device_config["TARGET_FALLBACK"] = "GPU,CPU";
    }
};

TEST_P(BatchResizeTest, batchTest) {
    auto fn_ptr = createSubgraph(_netDims);
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setPrecision(_inputPrecision);
    net.getInputsInfo().begin()->second->setLayout(_inputLayout);
    net.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(_resAlg);
    net.getInputsInfo().begin()->second->getPreProcess().setColorFormat(_colorFormat);

    auto execNet = ie->LoadNetwork(net, _device, device_config);
    auto req = execNet.CreateInferRequest();

    Blob::Ptr inputBlob;
    Blob::Ptr refBlob;

    // we use an image resized by openCV as a reference value.
    cv::InterpolationFlags cv_interpolation = (_resAlg == RESIZE_BILINEAR) ? cv::INTER_LINEAR : cv::INTER_AREA;

    std::vector<cv::Mat> resizedImgs(_imgs.size());
    for (size_t i = 0; i < _imgs.size(); ++i) {
        switch (_inputPrecision) {
            case Precision::FP32: {
                cv::Mat resizedImg_;
                _imgs[i].convertTo(resizedImg_, CV_32FC3);
                cv::resize(resizedImg_, resizedImgs[i], cv::Size(_netDims[3], _netDims[2]),
                    0, 0, cv_interpolation);
            }
            break;
            case Precision::U8: {
                cv::resize(_imgs[i], resizedImgs[i], cv::Size(_netDims[3], _netDims[2]),
                    0, 0, cv_interpolation);
            }
            break;
            default:
                IE_THROW()  << "Can't resize data of inconsistent precision: "
                                    << _inputPrecision;
        }
    }

    if (_doColorConversion) {
        for (auto& img : _imgs) {
                cv::cvtColor(img, img, toCvtColorType(_colorFormat));
        }
    }

    // set inputBlob to the whole batch
    switch (_inputPrecision) {
        case Precision::FP32: {
            inputBlob = img2Blob<Precision::FP32>(_imgs, _inputLayout);
        }
        break;
        case Precision::U8: {
            inputBlob = img2Blob<Precision::U8>(_imgs, _inputLayout);
        }
        break;
        default:
            IE_THROW()  << "Can't resize data of inconsistent precision: "
                                << _inputPrecision;
    }

    refBlob = img2Blob<Precision::FP32>(resizedImgs, Layout::NCHW);

    req.SetBlob(net.getInputsInfo().begin()->first, inputBlob);

    if (_isAsync) {
        req.StartAsync();
        req.Wait(InferRequest::WaitMode::RESULT_READY);
    } else {
        req.Infer();
    }

    Blob::Ptr outputBlob = req.GetBlob(net.getOutputsInfo().begin()->first);

    if (refBlob->size() != outputBlob->size()) {
        IE_THROW() << "reference and output blobs have different sizes!";
    }

    compare(*outputBlob, *refBlob, _threshold);
}

// Separate test fixture due to limited support of dynamic batching in plugins.
class DynamicBatchResizeTest : public BatchResizeTest {
protected:
    const size_t _max_batch_size = 10;
};

TEST_P(DynamicBatchResizeTest, dynamicBatchTest) {
    // use N = 1 for model generation (IR + weights). set maximum batch size (const value) to
    // CNNNetwork. set batch_size from _netDims to InferRequest as a dynamic batch value. overall,
    // this tests the usual case when IR provided doesn't depend on specific batch, while during
    // inference we want to vary the batch size according to input data available.
    auto batch_size = _netDims[0];
    _netDims[0] = 1;

    auto fn_ptr = createSubgraph(_netDims);
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setPrecision(_inputPrecision);
    net.getInputsInfo().begin()->second->setLayout(_inputLayout);
    net.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(_resAlg);
    net.getInputsInfo().begin()->second->getPreProcess().setColorFormat(_colorFormat);

    // enable dynamic batching and prepare for setting max batch limit.

    device_config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
    net.setBatchSize(_max_batch_size);

    auto execNet = ie->LoadNetwork(net, _device, device_config);
    auto req = execNet.CreateInferRequest();

    Blob::Ptr inputBlob;
    Blob::Ptr outputBlob;
    Blob::Ptr refBlob;

    // we use an image resized by OpenCV as a reference value.
    cv::InterpolationFlags cv_interpolation = (_resAlg == RESIZE_BILINEAR)
                                              ? cv::INTER_LINEAR : cv::INTER_AREA;

    for (int i = 0; i < _maxRepeat; ++i) {

    _imgs.clear();
    for (size_t j = 0; j < batch_size; ++j) {
        cv::Mat img(200, 200, CV_8UC3);
        cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));
        _imgs.emplace_back(img);
        if (i != 0) auxDownscale(_imgs.back(), cv_interpolation);
    }

    // fill in input images outside of current batch with random values. these should not occur in
    // output blob after inference.
    auto diff = _max_batch_size - batch_size;
    for (; diff > 0; --diff) {
        cv::Mat random(_imgs[0].size(), _imgs[0].type());
        cv::randn(random, cv::Scalar::all(127), cv::Scalar::all(40.f));
        _imgs.emplace_back(std::move(random));
    }

    // use identity matrices to initialize output blob and OpenCV-resized images to initialize
    // reference blob. we use specific init values for output to ensure that output is specifically
    // initialized and be able to consistently check part that is outside of current batch size.
    std::vector<cv::Mat> identityImgs(_imgs);
    std::vector<cv::Mat> resizedImgs(_imgs);
    for (size_t i = 0; i < batch_size; ++i) {
        switch (_inputPrecision) {
            case Precision::FP32: {
                cv::Mat resizedImg_;
                _imgs[i].convertTo(resizedImg_, CV_32FC3);
                cv::resize(resizedImg_, resizedImgs[i], cv::Size(_netDims[3], _netDims[2]),
                    0, 0, cv_interpolation);
                identityImgs[i] = cv::Mat::eye(cv::Size(_netDims[3], _netDims[2]),
                    resizedImg_.type());
            }
            break;
            case Precision::U8: {
                cv::resize(_imgs[i], resizedImgs[i], cv::Size(_netDims[3], _netDims[2]),
                    0, 0, cv_interpolation);
                identityImgs[i] = cv::Mat::eye(cv::Size(_netDims[3], _netDims[2]), _imgs[i].type());
            }
            break;
            default:
                IE_THROW()  << "Can't resize data of inconsistent precision: "
                                    << _inputPrecision;
        }
    }

    // update images that are outside of current batch: these remain unchaged after inference =>
    // resized == identity. If for some reason they're different, something changed them (in this
    // test, likely preprocessing) which must not happen.
    for (size_t i = batch_size; i < _imgs.size(); ++i) {
        resizedImgs[i] = cv::Mat::eye(cv::Size(_netDims[3], _netDims[2]), resizedImgs[0].type());
        identityImgs[i] = cv::Mat::eye(cv::Size(_netDims[3], _netDims[2]), resizedImgs[0].type());
    }

    if (_doColorConversion) {
        for (auto& img : _imgs) {
                cv::cvtColor(img, img, toCvtColorType(_colorFormat));
        }
    }

    // set inputBlob to the whole batch
    switch (_inputPrecision) {
        case Precision::FP32: {
            inputBlob = img2Blob<Precision::FP32>(_imgs, _inputLayout);
        }
        break;
        case Precision::U8: {
            inputBlob = img2Blob<Precision::U8>(_imgs, _inputLayout);
        }
        break;
        default:
            IE_THROW()  << "Can't resize data of inconsistent precision: "
                                << _inputPrecision;
    }

    outputBlob = img2Blob<Precision::FP32>(identityImgs, Layout::NCHW);
    refBlob = img2Blob<Precision::FP32>(resizedImgs, Layout::NCHW);

    req.SetBlob(net.getInputsInfo().begin()->first, inputBlob);
    req.SetBlob(net.getOutputsInfo().begin()->first, outputBlob);

    // Note: order of SetBlob and SetBatch matters! at the time of SetBlob, preprocessing is
    // initialized. If SetBatch is called before SetBlob, it may have no effect on preprocessing
    // because there's no preprocessing instances available yet.
    req.SetBatch(batch_size);
    if (_isAsync) {
        req.StartAsync();
        req.Wait(InferRequest::WaitMode::RESULT_READY);
    } else {
        req.Infer();
    }

    if (refBlob->size() != outputBlob->size()) {
        IE_THROW() << "reference and output blobs have different sizes!";
    }

    compare(*outputBlob, *refBlob, _threshold);

    }
}

class ReorderTest : public ResizeBase<std::string, std::pair<Layout, Layout>> {
protected:
    cv::Mat _img;
    Layout _networkLayout = Layout::ANY;

    void SetUp() override {
        ResizeBase<std::string, std::pair<Layout, Layout>>::SetUp();

        _device = std::get<0>(GetParam());
        std::string imgFile;
        std::pair<Precision, float> inPrcThresh;
        std::pair<Layout, Layout> layouts;
        std::tie(
                _netPrc,
                _netDims,
                inPrcThresh,
                layouts,
                _resAlg,
                _colorFormat,
                _cropRoi,
                _isAsync
        ) = std::get<1>(GetParam());

        _img = cv::Mat(300, 300, CV_8UC3);
        cv::randu(_img, cv::Scalar(0, 0, 0), cv::Scalar(255,255, 255));

        _inputPrecision = inPrcThresh.first;
        _threshold = inPrcThresh.second;

        _doColorConversion = _colorFormat != ColorFormat::RAW && _colorFormat != ColorFormat::BGR;

        std::tie(_inputLayout, _networkLayout) = layouts;

        if (_device == "HETERO")
            device_config["TARGET_FALLBACK"] = "GPU,CPU";
    }
};

TEST_P(ReorderTest, reorderTest) {
    auto fn_ptr = createSubgraph(_netDims);
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setPrecision(_inputPrecision);
    net.getInputsInfo().begin()->second->setLayout(_networkLayout);
    net.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(_resAlg);
    net.getInputsInfo().begin()->second->getPreProcess().setColorFormat(_colorFormat);

    auto execNet = ie->LoadNetwork(net, _device, device_config);
    auto req = execNet.CreateInferRequest();

    Blob::Ptr inputBlob;
    Blob::Ptr refBlob;

    for (int i = 0; i < _maxRepeat; ++i) {

    switch (_inputPrecision) {
        case Precision::FP32: {
            inputBlob = img2Blob<Precision::FP32>(_img, _inputLayout);
        }
        break;
        case Precision::U8: {
            inputBlob = img2Blob<Precision::U8>(_img, _inputLayout);
        }
        break;
        default:
            IE_THROW() << "Can't resize data of inconsistent precision: " << _inputPrecision;
    }

    refBlob = img2Blob<Precision::FP32>(_img, Layout::NCHW);

    req.SetBlob(net.getInputsInfo().begin()->first, inputBlob);

    if (_isAsync) {
        req.StartAsync();
        req.Wait(InferRequest::WaitMode::RESULT_READY);
    } else {
        req.Infer();
    }

    Blob::Ptr outputBlob = req.GetBlob(net.getOutputsInfo().begin()->first);

    if (refBlob->size() != outputBlob->size()) {
        IE_THROW() << "reference and output blobs have different sizes!";
    }

    compare(*outputBlob, *refBlob, _threshold);

    }
}

using nv12_test_params = std::tuple<
    std::string,                          // Plugin name
    std::tuple<
            Precision,                    // Network precision
            cv::Size,                     // Input image size
            SizeVector,                   // Net input sizes
            std::pair<Precision, float>,  // Input data precision and threshold
            ResizeAlgorithm,              // Resize algorithm kind
            ColorFormat,                  // Input color format kind
            ROI,                          // Cropped ROI coordinates
            bool                          // Infer modes: true = Async, false = Sync
    >
>;

class NV12ColorConvertTest : public Base<nv12_test_params> {
protected:
    cv::Size _inputSize;  // Input image size
    InferenceEngine::ROI _yRoi;
    InferenceEngine::ROI _uvRoi;

    void SetUp() override {
        Base<nv12_test_params>::SetUp();

        _device = std::get<0>(GetParam());
        std::pair<Precision, float> inPrcThresh;
        std::tie(
                _netPrc,
                _inputSize,
                _netDims,
                inPrcThresh,
                _resAlg,
                _colorFormat,
                _cropRoi,
                _isAsync
        ) = std::get<1>(GetParam());

        _inputPrecision = inPrcThresh.first;
        _threshold = inPrcThresh.second;

        _inputLayout = Layout::NCHW;

        if (_inputPrecision != Precision::U8) {
            IE_THROW() << "Specified input precision != Precision::U8";
        }

        _yRoi = _cropRoi;
        _uvRoi = _cropRoi / 2;

        if (_device == "HETERO")
            device_config["TARGET_FALLBACK"] = "GPU,CPU";
    }
};

TEST_P(NV12ColorConvertTest, NV12Test) {
    auto fn_ptr = createSubgraph(_netDims);
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setPrecision(_inputPrecision);
    net.getInputsInfo().begin()->second->setLayout(_inputLayout);
    net.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(_resAlg);
    net.getInputsInfo().begin()->second->getPreProcess().setColorFormat(_colorFormat);

    auto execNet = ie->LoadNetwork(net, _device, device_config);
    auto req = execNet.CreateInferRequest();

    // we use an image resized by openCV as a reference value.
    cv::InterpolationFlags cv_interpolation = (_resAlg == RESIZE_BILINEAR) ? cv::INTER_LINEAR : cv::INTER_AREA;

    for (int i = 0; i < _maxRepeat; ++i) {

    auto yRoi = _yRoi;
    auto uvRoi = _uvRoi;

    if (i != 0) {
        yRoi = auxDownscaledRoi();
        uvRoi = yRoi / 2;
    }

    cv::Mat yPlane(_inputSize, CV_MAKE_TYPE(CV_8U, 1)),
            uvPlane(cv::Size(_inputSize.width/2, _inputSize.height/2), CV_MAKE_TYPE(CV_8U, 2));

    cv::randn(yPlane, cv::Scalar::all(127), cv::Scalar::all(40.f));
    cv::randu(uvPlane, cv::Scalar::all(0), cv::Scalar::all(255));

    auto toRect = [] (const InferenceEngine::ROI& roi) {
        cv::Rect location;
        location.x = roi.posX;
        location.y = roi.posY;
        location.width = roi.sizeX;
        location.height = roi.sizeY;
        return location;
    };
    // Note: using 2 ROIs in case Y ROI has non even offset at the beginning: e.g. { .x = 25, ... },
    //       this way UV ROI will have x = 12! this is critical in case of ROI applied to BGR
    //       (converted from NV12) image, so ROIs are applied before conversion to be always
    //       compliant with IE code
    cv::Rect yLocation = toRect(yRoi);
    cv::Rect uvLocation = toRect(uvRoi);
    auto yPlaneCropped = yPlane(yLocation);
    auto uvPlaneCropped = uvPlane(uvLocation);

    cv::Mat refImg;  // converted and resized image
    cv::cvtColorTwoPlane(yPlaneCropped, uvPlaneCropped, refImg, cv::COLOR_YUV2BGR_NV12);
    cv::resize(refImg, refImg, cv::Size(_netDims[3], _netDims[2]), 0, 0, cv_interpolation);
    auto refBlob = img2Blob<Precision::FP32>(refImg, Layout::NCHW);

    auto yBlob = img2Blob<Precision::U8>(yPlane, NHWC);
    auto uvBlob = img2Blob<Precision::U8>(uvPlane, NHWC);
    Blob::Ptr inputBlob;

    if (i % 2)
    {
        // New way to create NV12 ROI

        auto nv12Blob = make_shared_blob<NV12Blob>(yBlob, uvBlob);
        inputBlob = nv12Blob->createROI(yRoi);
    }
    else
    {
        // Old way to create NV12 ROI

        // Note: Y and UV blobs for original data must always be "alive" until the end of the execution:
        //       ROI blobs do not own the data
        auto croppedYBlob = make_shared_blob(yBlob, yRoi);
        auto croppedUvBlob = make_shared_blob(uvBlob, uvRoi);
        inputBlob = make_shared_blob<NV12Blob>(croppedYBlob, croppedUvBlob);
    }

    req.SetBlob(net.getInputsInfo().begin()->first, inputBlob);

    if (_isAsync) {
        req.StartAsync();
        req.Wait(InferRequest::WaitMode::RESULT_READY);
    } else {
        req.Infer();
    }

    Blob::Ptr outputBlob = req.GetBlob(net.getOutputsInfo().begin()->first);

    if (refBlob->size() != outputBlob->size()) {
        IE_THROW() << "reference and output blobs have different sizes!";
    }

    compare(*outputBlob, *refBlob, _threshold);

    }
}

// multiple values macro wrapper
#define MULTI_VALUE(...) __VA_ARGS__

// sizes of the network to be tested
#define TESTED_DIMS(batch_size) \
    SizeVector({batch_size, 3, 200, 200}), \
    SizeVector({batch_size, 3, 300, 300}), \
    SizeVector({batch_size, 3, 400, 400}), \
    SizeVector({batch_size, 3, 300, 199}), \
    SizeVector({batch_size, 3, 199, 300})

// sizes of the network to be tested
#define TESTED_DIMS_SMALL(batch_size) \
    SizeVector({batch_size, 3, 200, 200}), \
    SizeVector({batch_size, 3, 400, 400})

#define COLOR_FORMATS_RAW \
    ColorFormat::RAW

#define COLOR_FORMATS_3CH \
    ColorFormat::BGR, ColorFormat::RGB

#define COLOR_FORMATS_4CH \
    ColorFormat::BGRX, ColorFormat::RGBX

// #define PLUGING_CASE(_plugin, _test, _params) \
//     INSTANTIATE_TEST_SUITE_P(_plugin##_run, _test, Combine(Values(#_plugin "Plugin"), _params) )

#define PLUGING_CASE_WITH_SUFFIX(_device, _suffix, _test, _params) \
    INSTANTIATE_TEST_SUITE_P(_device##_run##_suffix, _test, Combine(Values(#_device), _params) )

#endif  // USE_OPENCV
