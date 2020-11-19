// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_blob_stream.hpp"

#include <ie_blob.h>
#include <istream>

InferenceEngine::details::BlobStream::BlobBuffer::BlobBuffer(const InferenceEngine::Blob::CPtr& blob) {
    char* data = nullptr;
    std::streampos size;
    if (!blob) {
        size = 0;
    } else {
        data = blob->cbuffer().as<char*>();
        size = blob->byteSize();
    }
    setg(data, data, data + size);
}
InferenceEngine::details::BlobStream::BlobBuffer::~BlobBuffer() {}

std::streampos InferenceEngine::details::BlobStream::BlobBuffer::seekpos(std::streampos sp, std::ios_base::openmode which) {
    if (!(which & ios_base::in))
        return streampos(-1);
    if (sp < 0 || sp > egptr() - eback())
        return streampos(-1);
    setg(eback(), eback() + sp, egptr());
    return sp;
}
std::streampos InferenceEngine::details::BlobStream::BlobBuffer::seekoff(std::streamoff off, std::ios_base::seekdir way, std::ios_base::openmode which) {
    if (!(which & std::ios_base::in))
        return streampos(-1);
    switch (way) {
    default:
    case std::ios_base::beg:
        setg(eback(), eback() + off, egptr());
        break;
    case std::ios_base::cur:
        gbump(static_cast<int>(off));
        break;
    case std::ios_base::end:
        setg(eback(), egptr() + off, egptr());
        break;
    }
    return gptr() - eback();
}

InferenceEngine::Blob::CPtr InferenceEngine::details::BlobStream::getBlob() {
    return blob;
}

InferenceEngine::details::BlobStream::BlobStream(const InferenceEngine::Blob::CPtr& blob): buffer(blob), std::ios(0), std::istream(&buffer), blob(blob) {}

InferenceEngine::details::BlobStream::~BlobStream() {}
