// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <istream>

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(BlobStream): public std::istream {
private:
    class BlobBuffer: public std::streambuf {
    public:
        BlobBuffer(const Blob::CPtr& blob);
        ~BlobBuffer() override;
        std::streampos seekpos(std::streampos sp, std::ios_base::openmode which) override;
        std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way, std::ios_base::openmode which) override;
    };

    BlobBuffer buffer;
    Blob::CPtr blob;

public:
    BlobStream(const Blob::CPtr& blob);
    ~BlobStream() override;

    Blob::CPtr getBlob();
};


}  // namespace details
}  // namespace InferenceEngine
