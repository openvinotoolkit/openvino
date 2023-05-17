// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_preprocess_data.hpp"

#include <ie_input_info.hpp>
#include <memory>

#include "debug.h"
#include "ie_system_conf.h"

namespace InferenceEngine {

/**
 * @brief This class stores pre-process information for exact input
 */
class PreProcessData : public IPreProcessData {
    /**
     * @brief ROI blob.
     */
    Blob::Ptr _userBlob = nullptr;

public:
    void setRoiBlob(const Blob::Ptr& blob) override;

    Blob::Ptr getRoiBlob() const override;

    void execute(Blob::Ptr& preprocessedBlob, const PreProcessInfo& info, bool serial, int batchSize = -1) override;

    void isApplicable(const Blob::Ptr& src, const Blob::Ptr& dst) override;
};

void CreatePreProcessData(std::shared_ptr<IPreProcessData>& data) {
    data = std::make_shared<PreProcessData>();
}

void PreProcessData::setRoiBlob(const Blob::Ptr& blob) {
    _userBlob = blob;
}

Blob::Ptr PreProcessData::getRoiBlob() const {
    return _userBlob;
}

void PreProcessData::execute(Blob::Ptr& preprocessedBlob, const PreProcessInfo& info, bool serial, int batchSize) {
    IE_THROW() << "GAPI preprocess was removed!";
}

void PreProcessData::isApplicable(const Blob::Ptr& src, const Blob::Ptr& dst) {
    IE_THROW() << "GAPI preprocess was removed!";
}

}  // namespace InferenceEngine
