// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>

#include "ClassificationProcessor.hpp"
#include "Processor.hpp"

using InferenceEngine::details::InferenceEngineException;

ClassificationProcessor::ClassificationProcessor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, int flags_b,
        InferencePlugin plugin, CsvDumper& dumper, const std::string& flags_l,
        PreprocessingOptions preprocessingOptions, bool zeroBackground)
    : Processor(flags_m, flags_d, flags_i, flags_b, plugin, dumper, "Classification network", preprocessingOptions), zeroBackground(zeroBackground) {

    // Change path to labels file if necessary
    if (flags_l.empty()) {
        labelFileName = fileNameNoExt(modelFileName) + ".labels";
    } else {
        labelFileName = flags_l;
    }
}

ClassificationProcessor::ClassificationProcessor(const std::string& flags_m, const std::string& flags_d, const std::string& flags_i, int flags_b,
        InferencePlugin plugin, CsvDumper& dumper, const std::string& flags_l, bool zeroBackground)
    : ClassificationProcessor(flags_m, flags_d, flags_i, flags_b, plugin, dumper, flags_l,
            PreprocessingOptions(false, ResizeCropPolicy::ResizeThenCrop, 256, 256), zeroBackground) {
}

std::shared_ptr<Processor::InferenceMetrics> ClassificationProcessor::Process(bool stream_output) {
     slog::info << "Collecting labels" << slog::endl;
     ClassificationSetGenerator generator;
     try {
         generator.readLabels(labelFileName);
     } catch (InferenceEngine::details::InferenceEngineException& ex) {
         slog::warn << "Can't read labels file " << labelFileName << slog::endl;
         slog::warn << "Error: " << ex.what() << slog::endl;
     }

     auto validationMap = generator.getValidationMap(imagesPath);
     ImageDecoder decoder;

     // ----------------------------Do inference-------------------------------------------------------------
     slog::info << "Starting inference" << slog::endl;

     std::vector<int> expected(batch);
     std::vector<std::string> files(batch);

     ConsoleProgress progress(validationMap.size(), stream_output);

     ClassificationInferenceMetrics im;

     std::string firstInputName = this->inputInfo.begin()->first;
     std::string firstOutputName = this->outInfo.begin()->first;
     auto firstInputBlob = inferRequest.GetBlob(firstInputName);
     auto firstOutputBlob = inferRequest.GetBlob(firstOutputName);

     auto iter = validationMap.begin();
     while (iter != validationMap.end()) {
         size_t b = 0;
         int filesWatched = 0;
         for (; b < batch && iter != validationMap.end(); b++, iter++, filesWatched++) {
             expected[b] = iter->first;
             try {
                 decoder.insertIntoBlob(iter->second, b, *firstInputBlob, preprocessingOptions);
                 files[b] = iter->second;
             } catch (const InferenceEngineException& iex) {
                 slog::warn << "Can't read file " << iter->second << slog::endl;
                 slog::warn << "Error: " << iex.what() << slog::endl;
                 // Could be some non-image file in directory
                 b--;
                 continue;
             }
         }

         Infer(progress, filesWatched, im);

         std::vector<unsigned> results;
         auto firstOutputData = firstOutputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
         InferenceEngine::TopResults(TOP_COUNT, *firstOutputBlob, results);

         for (size_t i = 0; i < b; i++) {
             int expc = expected[i];
             if (zeroBackground) expc++;

             bool top1Scored = (static_cast<int>(results[0 + TOP_COUNT * i]) == expc);
             dumper << "\"" + files[i] + "\"" << top1Scored;
             if (top1Scored) im.top1Result++;
             for (int j = 0; j < TOP_COUNT; j++) {
                 unsigned classId = results[j + TOP_COUNT * i];
                 if (static_cast<int>(classId) == expc) {
                     im.topCountResult++;
                 }
                 dumper << classId << firstOutputData[classId + i * (firstOutputBlob->size() / batch)];
             }
             dumper.endLine();
             im.total++;
         }
     }
     progress.finish();

     return std::shared_ptr<Processor::InferenceMetrics>(new ClassificationInferenceMetrics(im));
}

void ClassificationProcessor::Report(const Processor::InferenceMetrics& im) {
    Processor::Report(im);
    if (im.nRuns > 0) {
        const ClassificationInferenceMetrics& cim = dynamic_cast<const ClassificationInferenceMetrics&>(im);

        cout << "Top1 accuracy: " << OUTPUT_FLOATING(100.0 * cim.top1Result / cim.total) << "% (" << cim.top1Result << " of "
                << cim.total << " images were detected correctly, top class is correct)" << "\n";
        cout << "Top5 accuracy: " << OUTPUT_FLOATING(100.0 * cim.topCountResult / cim.total) << "% (" << cim.topCountResult << " of "
            << cim.total << " images were detected correctly, top five classes contain required class)" << "\n";
    }
}

