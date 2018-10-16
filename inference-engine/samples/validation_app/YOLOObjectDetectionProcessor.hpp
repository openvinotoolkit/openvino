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
#include <list>
#include <vector>
#include <algorithm>

using namespace std;

class YOLOObjectDetectionProcessor : public ObjectDetectionProcessor {
private:
    /**
     * \brief This function analyses the YOLO net output for a single class
     * @param net_out - The output data
     * @param class_num - The class number
     * @return a list of found boxes
     */
    std::vector<DetectedObject> yoloNetParseOutput(const float *net_out, int class_num) {
        float threshold = 0.2f;         // The confidence threshold
        int C = 20;                     // classes
        int B = 2;                      // bounding boxes
        int S = 7;                      // cell size

        std::vector<DetectedObject> boxes;
        std::vector<DetectedObject> boxes_result;
        int SS = S * S;                 // number of grid cells 7*7 = 49
        // First 980 values corresponds to probabilities for each of the 20 classes for each grid cell.
        // These probabilities are conditioned on objects being present in each grid cell.
        int prob_size = SS * C;         // class probabilities 49 * 20 = 980
        // The next 98 values are confidence scores for 2 bounding boxes predicted by each grid cells.
        int conf_size = SS * B;         // 49*2 = 98 confidences for each grid cell

        const float *probs = &net_out[0];
        const float *confs = &net_out[prob_size];
        const float *cords = &net_out[prob_size + conf_size];     // 98*4 = 392 coords x, y, w, h

        for (int grid = 0; grid < SS; grid++) {
            int row = grid / S;
            int col = grid % S;
            for (int b = 0; b < B; b++) {
                int index = grid * B + b;
                int p_index = SS * C + grid * B + b;
                float scale = net_out[p_index];
                int box_index = SS * (C + B) + (grid * B + b) * 4;
                int objectType = class_num;

                float conf = confs[(grid * B + b)];
                float xc = (cords[(grid * B + b) * 4 + 0] + col) / S;
                float yc = (cords[(grid * B + b) * 4 + 1] + row) / S;
                float w = pow(cords[(grid * B + b) * 4 + 2], 2);
                float h = pow(cords[(grid * B + b) * 4 + 3], 2);
                int class_index = grid * C;
                float prob = probs[grid * C + class_num] * conf;

                DetectedObject bx(objectType, xc - w / 2, yc - h / 2, xc + w / 2,
                        yc + h / 2, prob);

                if (prob >= threshold) {
                    boxes.push_back(bx);
                }
            }
        }

        // Sorting the higher probabilities to the top
        sort(boxes.begin(), boxes.end(),
                [](const DetectedObject & a, const DetectedObject & b) -> bool {
                    return a.prob > b.prob;
                });

        // Filtering out overlapping boxes
        std::vector<bool> overlapped(boxes.size(), false);
        for (int i = 0; i < boxes.size(); i++) {
            if (overlapped[i])
                continue;

            DetectedObject box_i = boxes[i];
            for (int j = i + 1; j < boxes.size(); j++) {
                DetectedObject box_j = boxes[j];
                if (DetectedObject::ioU(box_i, box_j) >= 0.4) {
                    overlapped[j] = true;
                }
            }
        }

        for (int i = 0; i < boxes.size(); i++) {
            if (boxes[i].prob > 0.0f) {
                boxes_result.push_back(boxes[i]);
            }
        }
        return boxes_result;
    }

protected:
    std::map<std::string, std::list<DetectedObject>> processResult(std::vector<std::string> files) {
        std::map<std::string, std::list<DetectedObject>> detectedObjects;

        std::string firstOutputName = this->outInfo.begin()->first;
        const auto detectionOutArray = inferRequest.GetBlob(firstOutputName);
        const float *box = detectionOutArray->buffer().as<float*>();

        std::string file = *files.begin();
        for (int c = 0; c < 20; c++) {
            std::vector<DetectedObject> result = yoloNetParseOutput(box, c);
            detectedObjects[file].insert(detectedObjects[file].end(), result.begin(), result.end());
        }

        return detectedObjects;
    }

public:
    YOLOObjectDetectionProcessor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, const std::string& subdir, int flags_b,
            double threshold,
            InferencePlugin plugin, CsvDumper& dumper,
            const std::string& flags_a, const std::string& classes_list_file) :

                ObjectDetectionProcessor(flags_m, flags_d, flags_i, subdir, flags_b, threshold,
                        plugin, dumper, flags_a, classes_list_file, PreprocessingOptions(true, ResizeCropPolicy::Resize), false) { }
};
