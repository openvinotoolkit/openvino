// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <map>
#include <list>
#include <algorithm>
#include <memory>
#include <utility>

#include "ObjectDetectionProcessor.hpp"
#include "Processor.hpp"
#include "user_exception.hpp"

#include <samples/common.hpp>
#include <samples/slog.hpp>

using InferenceEngine::details::InferenceEngineException;

ObjectDetectionProcessor::ObjectDetectionProcessor(const std::string& flags_m, const std::string& flags_d,
        const std::string& flags_i, const std::string& subdir, int flags_b,
        double threshold, InferenceEngine::InferencePlugin plugin, CsvDumper& dumper,
        const std::string& flags_a, const std::string& classes_list_file, PreprocessingOptions preprocessingOptions, bool scaleProposalToInputSize)
            : Processor(flags_m, flags_d, flags_i, flags_b, plugin, dumper, "Object detection network", preprocessingOptions),
              annotationsPath(flags_a), subdir(subdir), threshold(threshold), scaleProposalToInputSize(scaleProposalToInputSize) {
    std::ifstream clf(classes_list_file);
    if (!clf) {
        throw UserException(1) <<  "Classes list file \"" << classes_list_file << "\" not found or inaccessible";
    }

    while (!clf.eof()) {
        std::string line;
        std::getline(clf, line, '\n');

        if (line != "") {
            istringstream lss(line);
            std::string id;
            lss >> id;
            int class_index = 0;
            lss >> class_index;

            classes.insert(std::pair<std::string, int>(id, class_index));
        }
    }
}

shared_ptr<Processor::InferenceMetrics> ObjectDetectionProcessor::Process(bool stream_output) {
    // Parsing PASCAL VOC2012 format
    VOCAnnotationParser vocAnnParser;
    slog::info << "Collecting VOC annotations from " << annotationsPath << slog::endl;
    VOCAnnotationCollector annCollector(annotationsPath);
    slog::info << annCollector.annotations().size() << " annotations collected" << slog::endl;

    if (annCollector.annotations().size() == 0) {
        ObjectDetectionInferenceMetrics emptyIM(this->threshold);

        return std::shared_ptr<InferenceMetrics>(new ObjectDetectionInferenceMetrics(emptyIM));
    }

    // Getting desired results from annotations
    std::map<std::string, ImageDescription> desiredForFiles;

    for (auto& ann : annCollector.annotations()) {
        std::list<DetectedObject> dobList;
        for (auto& obj : ann.objects) {
            DetectedObject dob(classes[obj.name], static_cast<float>(obj.bndbox.xmin),
                static_cast<float>(obj.bndbox.ymin), static_cast<float>(obj.bndbox.xmax),
                static_cast<float>(obj.bndbox.ymax), 1.0f, obj.difficult != 0);
            dobList.push_back(dob);
        }
        ImageDescription id(dobList);
        desiredForFiles.insert(std::pair<std::string, ImageDescription>(ann.folder + "/" + (!subdir.empty() ? subdir + "/" : "") + ann.filename, id));
    }

    for (auto & item : outInfo) {
        DataPtr outputData = item.second;
        if (!outputData) {
            throw std::logic_error("output data pointer is not valid");
        }
    }
    // -----------------------------------------------------------------------------------------------------

    // ----------------------------Do inference-------------------------------------------------------------
    slog::info << "Starting inference" << slog::endl;

    std::vector<VOCAnnotation> expected(batch);

    ConsoleProgress progress(annCollector.annotations().size(), stream_output);

    ObjectDetectionInferenceMetrics im(threshold);

    vector<VOCAnnotation>::const_iterator iter = annCollector.annotations().begin();

    std::map<std::string, ImageDescription> scaledDesiredForFiles;

    std::string firstInputName = this->inputInfo.begin()->first;
    auto firstInputBlob = inferRequest.GetBlob(firstInputName);

    while (iter != annCollector.annotations().end()) {
        std::vector<std::string> files;
        size_t b = 0;

        int filesWatched = 0;
        for (; b < batch && iter != annCollector.annotations().end(); b++, iter++, filesWatched++) {
            expected[b] = *iter;
            string filename = iter->folder + "/" + (!subdir.empty() ? subdir + "/" : "") + iter->filename;
            try {
                float scale_x, scale_y;

                scale_x = 1.0f / iter->size.width;  // orig_size.width;
                scale_y = 1.0f / iter->size.height;  // orig_size.height;

                if (scaleProposalToInputSize) {
                    scale_x *= firstInputBlob->dims()[0];
                    scale_y *= firstInputBlob->dims()[1];
                }

                // Scaling the desired result (taken from the annotation) to the network size
                scaledDesiredForFiles.insert(std::pair<std::string, ImageDescription>(filename, desiredForFiles.at(filename).scale(scale_x, scale_y)));

                files.push_back(filename);
            } catch (const InferenceEngineException& iex) {
                slog::warn << "Can't read file " << this->imagesPath + "/" + filename << slog::endl;
                slog::warn << "Error: " << iex.what() << slog::endl;
                // Could be some non-image file in directory
                b--;
                continue;
            }
        }

        if (files.size() == batch) {
            // Infer model
            Infer(progress, filesWatched, im);

            // Processing the inference result
            std::map<std::string, std::list<DetectedObject>> detectedObjects = processResult(files);

            // Calculating similarity
            //
            for (size_t b = 0; b < files.size(); b++) {
                ImageDescription result(detectedObjects[files[b]]);
                im.apc.consumeImage(result, scaledDesiredForFiles.at(files[b]));
            }
        }
    }
    progress.finish();

    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Postprocess output blobs--------------------------------------------------
    slog::info << "Processing output blobs" << slog::endl;

    return std::shared_ptr<InferenceMetrics>(new ObjectDetectionInferenceMetrics(im));
}

void ObjectDetectionProcessor::Report(const Processor::InferenceMetrics& im) {
    const ObjectDetectionInferenceMetrics& odim = dynamic_cast<const ObjectDetectionInferenceMetrics&>(im);
    Processor::Report(im);
    if (im.nRuns > 0) {
        std::map<int, double> appc = odim.apc.calculateAveragePrecisionPerClass();

        std::cout << "Average precision per class table: " << std::endl << std::endl;
        std::cout << "Class\tAP" << std::endl;

        double mAP = 0;
        for (auto i : appc) {
            std::cout << std::fixed << std::setprecision(3) << i.first << "\t" << i.second << std::endl;
            mAP += i.second;
        }
        mAP /= appc.size();
        std::cout << std::endl << std::fixed << std::setprecision(4) << "Mean Average Precision (mAP): " << mAP << std::endl;
    }
}
