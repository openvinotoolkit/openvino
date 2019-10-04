// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <vector>
#include <string>
#include "ie_const_infer_impl.hpp"

using namespace InferenceEngine;
using namespace ShapeInfer;

void ConstInferImpl::infer(const std::vector<Blob::CPtr>& inData,
                           const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs,
                           std::vector<Blob::Ptr>& outData) {
    std::string errorPrefix = "Ref infer error for Layer with `" + _type + "` type: ";
    if (outData.empty()) THROW_IE_EXCEPTION << errorPrefix + "output data is empty";
    for (auto const& data : outData) {
        if (data->buffer() == nullptr)
            THROW_IE_EXCEPTION << errorPrefix + "output data is not allocated";
    }
    // TODO: check for direct (NCHW, NCH, NC) and FP32
    inferImpl(inData, params, blobs, outData);
}

