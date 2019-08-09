// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>
#include "ie_blob.h"

#include "PreprocessingOptions.hpp"

using namespace InferenceEngine;

class ImageDecoder {
public:
    /**
     * @brief Load single image to blob
     * @param name - image file name
     * @param blob - blob object to load image data to
     * @return original image sizes
     */
    static cv::Size loadToBlob(const std::string& name, Blob& blob, const PreprocessingOptions& preprocessingOptions);

    /**
     * @brief Load a list of images to blob
     * @param names - list of images filenames
     * @param blob - blob object to load images data to
     * @return original image size
     */
    static std::map<std::string, cv::Size> loadToBlob(const std::vector<std::string>& names, Blob& blob, const PreprocessingOptions& preprocessingOptions);

    /**
     * @brief Insert image data to blob at specified batch position.
     *        Does no checks if blob has sufficient space
     * @param name - image file name
     * @param batch_pos - batch position image should be loaded to
     * @param blob - blob object to load image data to
     * @return original image size
     */
    static cv::Size insertIntoBlob(const std::string& name, int batch_pos, Blob& blob, const PreprocessingOptions& preprocessingOptions);
};
