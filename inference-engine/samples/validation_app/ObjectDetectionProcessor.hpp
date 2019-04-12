// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

    shared_ptr<InferenceMetrics> Process(bool stream_output);
    virtual void Report(const InferenceMetrics& im);
    virtual ~ObjectDetectionProcessor() {}
};
