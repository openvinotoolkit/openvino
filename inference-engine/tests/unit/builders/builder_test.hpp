// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>
#include <ie_builders.hpp>
#include <blob_factory.hpp>

#include "tests_common.hpp"


class BuilderTestCommon : public TestsCommon {
public:
    InferenceEngine::Blob::Ptr generateBlob(InferenceEngine::Precision precision,
                                            InferenceEngine::SizeVector dims, InferenceEngine::Layout layout) {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(InferenceEngine::TensorDesc(precision, dims, layout));
        blob->allocate();
        fill_data(blob);
        return blob;
    }

    template<class T>
    InferenceEngine::Blob::Ptr generateBlob(InferenceEngine::Precision precision,
                                            InferenceEngine::SizeVector dims, InferenceEngine::Layout layout,
                                            std::vector<T> data) {
        auto blob = generateBlob(precision, dims, layout);
        auto *blbData = blob->buffer().as<T *>();
        for (size_t i = 0; i < data.size(); i++) {
            blbData[i] = data[i];
        }
        return blob;
    }
};