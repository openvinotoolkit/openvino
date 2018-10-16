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
#include <string>
#include <memory>

#include "classification_set_generator.hpp"
#include "Processor.hpp"

using namespace std;

class ClassificationProcessor : public Processor {
    const int TOP_COUNT = 5;

    struct ClassificationInferenceMetrics : public InferenceMetrics {
    public:
        int top1Result = 0;
        int topCountResult = 0;
        int total = 0;
    };

protected:
    std::string labelFileName;
    bool zeroBackground;
public:
    ClassificationProcessor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, int flags_b,
            InferenceEngine::InferencePlugin plugin, CsvDumper& dumper, const std::string& flags_l,
            PreprocessingOptions preprocessingOptions, bool zeroBackground);
    ClassificationProcessor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, int flags_b,
            InferenceEngine::InferencePlugin plugin, CsvDumper& dumper, const std::string& flags_l, bool zeroBackground);

    std::shared_ptr<InferenceMetrics> Process();
    virtual void Report(const InferenceMetrics& im);
    virtual ~ClassificationProcessor() { }
};
