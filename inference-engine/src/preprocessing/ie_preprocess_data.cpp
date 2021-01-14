// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_preprocess_gapi.hpp"
#include "ie_system_conf.h"
#include "ie_preprocess_data.hpp"
#include "ie_preprocess_itt.hpp"

#include "debug.h"
#include <ie_input_info.hpp>

#include <memory>

namespace InferenceEngine {

/**
 * @brief This class stores pre-process information for exact input
 */
class PreProcessData : public IPreProcessData {
    /**
     * @brief ROI blob.
     */
    Blob::Ptr _userBlob = nullptr;

    /**
     * @brief Pointer-to-implementation (PIMPL) hiding preprocessing implementation details.
     * BEWARE! Will be shared among copies!
     */
    std::shared_ptr<PreprocEngine> _preproc;

public:
    void setRoiBlob(const Blob::Ptr &blob) override;

    Blob::Ptr getRoiBlob() const override;

    void execute(Blob::Ptr &preprocessedBlob, const PreProcessInfo &info, bool serial, int batchSize = -1) override;

    void Release() noexcept override;

    void isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst) override;
};

StatusCode CreatePreProcessData(IPreProcessData *& data, ResponseDesc * /*resp*/) noexcept {
    data = new PreProcessData();
    return StatusCode::OK;
}

void PreProcessData::Release() noexcept {
    delete this;
}

void PreProcessData::setRoiBlob(const Blob::Ptr &blob) {
    _userBlob = blob;
}

Blob::Ptr PreProcessData::getRoiBlob() const {
    return _userBlob;
}

void PreProcessData::execute(Blob::Ptr &preprocessedBlob, const PreProcessInfo &info, bool serial,
        int batchSize) {
    OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, "Preprocessing");

    auto algorithm = info.getResizeAlgorithm();
    auto fmt = info.getColorFormat();

    if (_userBlob == nullptr || preprocessedBlob == nullptr) {
        THROW_IE_EXCEPTION << "Input pre-processing is called with null " << (_userBlob == nullptr ? "_userBlob" : "preprocessedBlob");
    }

    batchSize = PreprocEngine::getCorrectBatchSize(batchSize, _userBlob);

    if (!_preproc) {
        _preproc.reset(new PreprocEngine);
    }

    _preproc->preprocessWithGAPI(_userBlob, preprocessedBlob, algorithm, fmt, serial, batchSize);
}

void PreProcessData::isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst) {

    PreprocEngine::checkApplicabilityGAPI(src, dst);
}

}  // namespace InferenceEngine
