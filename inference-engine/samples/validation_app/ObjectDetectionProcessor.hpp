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

#include "Processor.hpp"

#include "VOCAnnotationParser.hpp"

using namespace std;

class ObjectDetectionProcessor : public Processor {
public:
    struct ObjectDetectionInferenceMetrics : public InferenceMetrics {
    public:
        AveragePrecisionCalculator apc;

        explicit ObjectDetectionInferenceMetrics(double threshold) : apc(threshold) { }
    };

protected:
    std::string annotationsPath;
    std::string subdir;
    std::map<std::string, int> classes;
    double threshold;

    bool scaleProposalToInputSize;

    virtual std::map<std::string, std::list<DetectedObject>> processResult(std::vector<std::string> files) = 0;

public:
    ObjectDetectionProcessor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, const std::string& subdir, int flags_b,
            double threshold,
            InferenceEngine::InferencePlugin plugin, CsvDumper& dumper,
            const std::string& flags_a, const std::string& classes_list_file, PreprocessingOptions preprocessingOptions, bool scaleSizeToInputSize);

    shared_ptr<InferenceMetrics> Process();
    virtual void Report(const InferenceMetrics& im);
    virtual ~ObjectDetectionProcessor() {}
};
