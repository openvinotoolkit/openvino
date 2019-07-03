// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

    std::shared_ptr<InferenceMetrics> Process(bool stream_output);
    virtual void Report(const InferenceMetrics& im);
    virtual ~ClassificationProcessor() { }
};
