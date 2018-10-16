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

#pragma once

#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <utility>

#include "ObjectDetectionProcessor.hpp"

using namespace std;

class SSDObjectDetectionProcessor : public ObjectDetectionProcessor {
protected:
    std::map<std::string, std::list<DetectedObject>> processResult(std::vector<std::string> files) {
        std::map<std::string, std::list<DetectedObject>> detectedObjects;

        std::string firstOutputName = this->outInfo.begin()->first;
        const auto detectionOutArray = inferRequest.GetBlob(firstOutputName);
        const float *box = detectionOutArray->buffer().as<float*>();

        const int maxProposalCount = outputDims[1];
        const int objectSize = outputDims[0];

        for (int b = 0; b < batch; b++) {
            string fn = files[b];
            std::list<DetectedObject> dr = std::list<DetectedObject>();
            detectedObjects.insert(std::pair<std::string, std::list<DetectedObject>>(fn, dr));
        }

        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = box[i * objectSize + 0];
            float label = box[i * objectSize + 1];
            float confidence = box[i * objectSize + 2];
            float xmin = box[i * objectSize + 3] * inputDims[0];
            float ymin = box[i * objectSize + 4] * inputDims[1];
            float xmax = box[i * objectSize + 5] * inputDims[0];
            float ymax = box[i * objectSize + 6] * inputDims[1];

            if (image_id < 0 /* better than check == -1 */) {
                break;  // Finish
            }

            detectedObjects[files[image_id]].push_back(DetectedObject(label, xmin, ymin, xmax, ymax, confidence));
        }

        return detectedObjects;
    }

public:
    SSDObjectDetectionProcessor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, const std::string& subdir, int flags_b,
            double threshold,
            InferencePlugin plugin, CsvDumper& dumper,
            const std::string& flags_a, const std::string& classes_list_file) :

                ObjectDetectionProcessor(flags_m, flags_d, flags_i, subdir, flags_b, threshold,
                        plugin, dumper, flags_a, classes_list_file, PreprocessingOptions(false, ResizeCropPolicy::Resize), true) { }
};
