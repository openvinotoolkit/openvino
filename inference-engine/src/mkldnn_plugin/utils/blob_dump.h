//
// Copyright 2016-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "ie_blob.h"

#include <string>

namespace MKLDNNPlugin {

/**
 * Utility class to dump blob contant in plain format.
 * Every layout information will be lost.
 *
 * In case of low precision blob it allow to store
 * with using scaling factors per channel.
 * NB! Channel is a second dimension for all blob types.
 */
class BlobDumper {
    InferenceEngine::Blob::Ptr _blob;
    InferenceEngine::Blob::Ptr _scales;

public:
    BlobDumper() = default;
    BlobDumper(const BlobDumper&) = default;
    BlobDumper& operator = (BlobDumper&&) = default;

    explicit BlobDumper(const InferenceEngine::Blob::Ptr blob):_blob(blob) {}

    static BlobDumper read(const std::string &file_path);
    static BlobDumper read(std::istream &stream);

    void dump(const std::string &file_path);
    void dump(std::ostream &stream);

    void dumpAsTxt(const std::string file_path);
    void dumpAsTxt(std::ostream &stream);

    BlobDumper& withScales(InferenceEngine::Blob::Ptr scales);
    BlobDumper& withoutScales();

    const InferenceEngine::Blob::Ptr& getScales() const;

    InferenceEngine::Blob::Ptr get();
    InferenceEngine::Blob::Ptr getRealValue();
};

}  // namespace MKLDNNPlugin
