/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <utility>

#include "image_decoder.hpp"
#include "details/ie_exception.hpp"
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int getLoadModeForChannels(int channels, int base) {
    switch (channels) {
    case 1:
        return base | CV_LOAD_IMAGE_GRAYSCALE;
    case 3:
        return base | CV_LOAD_IMAGE_COLOR;
    }
    return base | CV_LOAD_IMAGE_UNCHANGED;
}

template <class T>
cv::Size addToBlob(std::string name, int batch_pos, Blob& blob, PreprocessingOptions preprocessingOptions) {
    SizeVector blobSize = blob.dims();
    int width = static_cast<int>(blobSize[0]);
    int height = static_cast<int>(blobSize[1]);
    int channels = static_cast<int>(blobSize[2]);
    T* blob_data = static_cast<T*>(blob.buffer());
    Mat orig_image, result_image;
    int loadMode = getLoadModeForChannels(channels, 0);

    std::string tryName = name;

    // TODO This is a dirty hack to support VOC2007 (where no file extension is put into annotation).
    //      Rewrite.
    if (name.find('.') == -1) tryName = name + ".JPEG";

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

    float scaleFactor = preprocessingOptions.scaleValuesTo01 ? 255.0 : 1.0;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                blob_data[batch_pos * channels * width * height + c * width * height + h * width + w] =
                    static_cast<T>(result_image.at<cv::Vec3b>(h, w)[c] / scaleFactor);
            }
        }
    }

    return res;
}

std::map<std::string, cv::Size> convertToBlob(std::vector<std::string> names, int batch_pos, Blob& blob, PreprocessingOptions preprocessingOptions) {
    if (blob.buffer() == nullptr) {
        THROW_IE_EXCEPTION << "Blob was not allocated";
    }

    std::function<cv::Size(std::string, int, Blob&, PreprocessingOptions)> add_func;

    switch (blob.precision()) {
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
    for (int b = 0; b < names.size(); b++) {
        std::string name = names[b];
        Size orig_size = add_func(name, batch_pos + b, blob, preprocessingOptions);
        res.insert(std::pair<std::string, Size>(name, orig_size));
    }

    return res;
}

Size ImageDecoder::loadToBlob(std::string name, Blob& blob, PreprocessingOptions preprocessingOptions) {
    std::vector<std::string> names = { name };
    return loadToBlob(names, blob, preprocessingOptions).at(name);
}

std::map<std::string, cv::Size> ImageDecoder::loadToBlob(std::vector<std::string> names, Blob& blob, PreprocessingOptions preprocessingOptions) {
    return convertToBlob(names, 0, blob, preprocessingOptions);
}

Size ImageDecoder::insertIntoBlob(std::string name, int batch_pos, Blob& blob, PreprocessingOptions preprocessingOptions) {
    return convertToBlob({ name }, batch_pos, blob, preprocessingOptions).at(name);
}
