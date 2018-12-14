// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <limits>
#include <string>
#include <memory>

#include <samples/common.hpp>

#include "inference_engine.hpp"

#include "csv_dumper.hpp"
#include "image_decoder.hpp"
#include "console_progress.hpp"

using namespace std;

#define OUTPUT_FLOATING(val) std::fixed << std::setprecision(2) << val

class Processor {
public:
    struct InferenceMetrics {
        int nRuns = 0;
        double minDuration = std::numeric_limits<double>::max();
        double maxDuration = 0;
        double totalTime = 0;

        virtual ~InferenceMetrics() { }  // Type has to be polymorphic
    };

protected:
    std::string modelFileName;
    std::string targetDevice;
    std::string imagesPath;
    int batch;
    InferenceEngine::InferRequest inferRequest;
    InferenceEngine::InputsDataMap inputInfo;
    InferenceEngine::OutputsDataMap outInfo;
    InferenceEngine::CNNNetReader networkReader;
    InferenceEngine::SizeVector inputDims;
    InferenceEngine::SizeVector outputDims;
    double loadDuration;
    PreprocessingOptions preprocessingOptions;

    CsvDumper& dumper;
    InferencePlugin plugin;

    std::string approach;

    double Infer(ConsoleProgress& progress, int filesWatched, InferenceMetrics& im);

public:
    Processor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, int flags_b,
            InferenceEngine::InferencePlugin plugin, CsvDumper& dumper, const std::string& approach, PreprocessingOptions preprocessingOptions);

    virtual shared_ptr<InferenceMetrics> Process() = 0;
    virtual void Report(const InferenceMetrics& im) {
        double averageTime = im.totalTime / im.nRuns;

        slog::info << "Inference report:\n";
        slog::info << "\tNetwork load time: " << loadDuration << "ms" << "\n";
        slog::info << "\tModel: " << modelFileName << "\n";
        slog::info << "\tModel Precision: " << networkReader.getNetwork().getPrecision().name() << "\n";
        slog::info << "\tBatch size: " << batch << "\n";
        slog::info << "\tValidation dataset: " << imagesPath << "\n";
        slog::info << "\tValidation approach: " << approach;
        slog::info << slog::endl;

        if (im.nRuns > 0) {
            slog::info << "Average infer time (ms): " << averageTime << " (" << OUTPUT_FLOATING(1000.0 / (averageTime / batch))
                    << " images per second with batch size = " << batch << ")" << slog::endl;
        } else {
            slog::warn << "No images processed" << slog::endl;
        }
    }

    virtual ~Processor() {}
};
