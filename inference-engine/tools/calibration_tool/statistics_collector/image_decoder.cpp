// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <samples/common.hpp>
#include <utility>

#include "details/ie_exception.hpp"
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "image_decoder.hpp"

using namespace cv;

int getLoadModeForChannels(int channels, int base) {
    switch (channels) {
    case 1:
        return base | IMREAD_GRAYSCALE;
    case 3:
        return base | IMREAD_COLOR;
    }
    return IMREAD_UNCHANGED;
}

template <class T>
cv::Size addToBlob(const std::string& name, int batch_pos, Blob& blob, const PreprocessingOptions& preprocessingOptions) {
    const TensorDesc& blobDesc = blob.getTensorDesc();
    const size_t width = getTensorWidth(blobDesc);
    const size_t height = getTensorHeight(blobDesc);
    const size_t channels = getTensorChannels(blobDesc);
    T* blob_data = static_cast<T*>(blob.buffer());
    Mat orig_image, result_image;
    int loadMode = getLoadModeForChannels(channels, 0);

    std::string tryName = name;

    // TODO This is a dirty hack to support VOC2007 (where no file extension is put into annotation).
    //      Rewrite.
    if (name.find('.') == std::string::npos) tryName = name + ".JPEG";

    orig_image = imread(tryName, loadMode);

    if (orig_image.empty()) {
        THROW_IE_EXCEPTION << "Cannot open image file: " << tryName;
    }

    // Preprocessing the image
    Size res = orig_image.size();

    if (preprocessingOptions.resizeCropPolicy == ResizeCropPolicy::Resize) {
        cv::resize(orig_image, result_image, Size(width, height));
    } else if (preprocessingOptions.resizeCropPolicy == ResizeCropPolicy::ResizeThenCrop) {
        Mat resized_image;

        cv::resize(orig_image, resized_image, Size(preprocessingOptions.resizeBeforeCropX, preprocessingOptions.resizeBeforeCropY));

        size_t cx = preprocessingOptions.resizeBeforeCropX / 2;
        size_t cy = preprocessingOptions.resizeBeforeCropY / 2;

        cv::Rect cropRect(cx - width / 2, cy - height / 2, width, height);
        result_image = resized_image(cropRect);
    } else if (preprocessingOptions.resizeCropPolicy == ResizeCropPolicy::DoNothing) {
        // No image preprocessing to be done here
        result_image = orig_image;
    } else {
        THROW_IE_EXCEPTION << "Unsupported ResizeCropPolicy value";
    }

    float scaleFactor = preprocessingOptions.scaleValuesTo01 ? 255.0f : 1.0f;

    size_t WH = width * height;
    size_t BCWH = batch_pos * channels * WH;
    const unsigned char* data = result_image.data;
    if (data == nullptr)
        THROW_IE_EXCEPTION << "Empty image " << name;
    const size_t CW = channels * width;
    for (size_t c = 0lu; c < channels; c++) {
        size_t c_BCWH = c * WH + BCWH;
        for (size_t h = 0lu; h < height; h++) {
            size_t h_c_BCWH = h * width + c_BCWH;
            size_t c_h_CW = CW * h + c;
            for (size_t w = 0lu; w < width; w++) {
                blob_data[h_c_BCWH + w] =
                    static_cast<T>(data[c_h_CW + w * channels] / scaleFactor);
            }
        }
    }

    return res;
}

std::map<std::string, cv::Size> convertToBlob(const std::vector<std::string>& names,
        int batch_pos, Blob& blob, const PreprocessingOptions& preprocessingOptions) {
    if (blob.buffer() == nullptr) {
        THROW_IE_EXCEPTION << "Blob was not allocated";
    }

    std::function<cv::Size(std::string, int, Blob&, PreprocessingOptions)> add_func;

    switch (blob.getTensorDesc().getPrecision()) {
    case Precision::FP32:
        add_func = &addToBlob<float>;
        break;
    case Precision::FP16:
    case Precision::Q78:
    case Precision::I16:
    case Precision::U16:
        add_func = &addToBlob<short>;
        break;
    default:
        add_func = &addToBlob<uint8_t>;
    }

    std::map<std::string, Size> res;
    for (size_t b = 0; b < names.size(); b++) {
        const std::string& name = names[b];
        Size orig_size = add_func(name, batch_pos + b, blob, preprocessingOptions);
        res.insert(std::pair<std::string, Size>(name, orig_size));
    }

    return res;
}

Size ImageDecoder::loadToBlob(const std::string& name, Blob& blob, const PreprocessingOptions& preprocessingOptions) {
    std::vector<std::string> names = { name };
    return loadToBlob(names, blob, preprocessingOptions).at(name);
}

std::map<std::string, cv::Size> ImageDecoder::loadToBlob(const std::vector<std::string>& names, Blob& blob, const PreprocessingOptions& preprocessingOptions) {
    return convertToBlob(names, 0, blob, preprocessingOptions);
}

Size ImageDecoder::insertIntoBlob(const std::string& name, int batch_pos, Blob& blob, const PreprocessingOptions& preprocessingOptions) {
    return convertToBlob({ name }, batch_pos, blob, preprocessingOptions).at(name);
}
