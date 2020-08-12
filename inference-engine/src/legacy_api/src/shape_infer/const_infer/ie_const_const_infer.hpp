// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <legacy/ie_layers.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for TBD layer
 */
class ConstConstInfer : public ConstInferImpl {
public:
    explicit ConstConstInfer(const std::string& type): ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        auto it = blobs.find("custom");
        if (it == blobs.end()) THROW_IE_EXCEPTION << "Missed `custom` blob";
        // TODO: copy instead of putting pointer?
        outData[0] = (*it).second;
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
