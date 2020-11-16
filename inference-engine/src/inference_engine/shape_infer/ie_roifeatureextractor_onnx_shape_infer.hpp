// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 * @brief Implementation of Shape inference for ExperimentalDetectronROIFeatureExtractor layer
 */
class ExperimentalDetectronROIFeatureExtractorShapeProp : public BuiltInShapeInferImpl {
protected:
    const int ROIS = 0;
    const int FEATMAPS = 1;

public:
    explicit ExperimentalDetectronROIFeatureExtractorShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        size_t rois_num = inShapes.at(ROIS).at(0);
        size_t channels_num = inShapes.at(FEATMAPS).at(1);
        size_t output_size = static_cast<size_t>(GetParamAsInt("output_size", params));
        outShapes.push_back({rois_num, channels_num, output_size, output_size});

        auto num_outputs = GetParamAsUInt("num_outputs", params);
        if (num_outputs > 2) THROW_IE_EXCEPTION << "Incorrect value num_outputs: " << num_outputs;
        if (num_outputs == 2) {
            outShapes.push_back({rois_num, 4});
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
