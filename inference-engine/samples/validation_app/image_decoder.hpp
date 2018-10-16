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

#pragma once

#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>
#include "ie_blob.h"

#include "PreprocessingOptions.hpp"

using namespace cv;
using namespace InferenceEngine;

class ImageDecoder {
public:
    /**
     * @brief Load single image to blob
     * @param name - image file name
     * @param blob - blob object to load image data to
     * @return original image sizes
     */
    Size loadToBlob(std::string name, Blob& blob, PreprocessingOptions preprocessingOptions);

    /**
     * @brief Load a list of images to blob
     * @param names - list of images filenames
     * @param blob - blob object to load images data to
     * @return original image size
     */
    std::map<std::string, cv::Size> loadToBlob(std::vector<std::string> names, Blob& blob, PreprocessingOptions preprocessingOptions);

    /**
     * @brief Insert image data to blob at specified batch position.
     *        Does no checks if blob has sufficient space
     * @param name - image file name
     * @param batch_pos - batch position image should be loaded to
     * @param blob - blob object to load image data to
     * @return original image size
     */
    Size insertIntoBlob(std::string name, int batch_pos, Blob& blob, PreprocessingOptions preprocessingOptions);
};
