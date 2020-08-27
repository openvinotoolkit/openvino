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
*@brief Implementation of Shape inference for ExperimentalDetectronDetectionOutput layer
*/
class ExperimentalDetectronDetectionOutputShapeProp : public BuiltInShapeInferImpl {
protected:
    const int ROIS = 0;
    const int FEATMAPS = 1;

public:
    explicit ExperimentalDetectronDetectionOutputShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);

        auto rois_num = cnnLayer.GetParamAsUInt("max_detections_per_image");
        outShapes.push_back({rois_num, 4});

        auto num_outputs = cnnLayer.GetParamAsUInt("num_outputs");
        if (num_outputs > 3) THROW_IE_EXCEPTION << "Incorrect value num_outputs: " << num_outputs;
        if (num_outputs >= 2) {
            outShapes.push_back({rois_num});
        }
        if (num_outputs == 3) {
            outShapes.push_back({rois_num});
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
