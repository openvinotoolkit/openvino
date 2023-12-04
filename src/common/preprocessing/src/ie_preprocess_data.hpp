// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <memory>

#include "openvino/runtime/common.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

#include <ie_blob.h>
#include <file_utils.h>
#include <ie_preprocess.hpp>
#include "ie_version.hpp"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START
/**
 * @brief This class stores pre-process information for exact input
 */
class IPreProcessData : public std::enable_shared_from_this<IPreProcessData> {
public:
    /**
     * @brief Sets ROI blob to be resized and placed to the default input blob during pre-processing.
     * @param blob ROI blob.
     */
    virtual void setRoiBlob(const Blob::Ptr &blob) = 0;

    /**
     * @brief Gets pointer to the ROI blob used for a given input.
     * @return Blob pointer.
     */
    virtual Blob::Ptr getRoiBlob() const = 0;

    /**
     * @brief Executes input pre-processing with a given pre-processing information.
     * @param outBlob pre-processed output blob to be used for inference.
     * @param info pre-processing info that specifies resize algorithm and color format.
     * @param serial disable OpenMP threading if the value set to true.
     * @param batchSize batch size for pre-processing.
     */
    virtual void execute(Blob::Ptr &preprocessedBlob, const PreProcessInfo& info, bool serial, int batchSize = -1) = 0;

    virtual void isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst) = 0;

protected:
    virtual ~IPreProcessData() = default;
};

OPENVINO_PLUGIN_API void CreatePreProcessData(std::shared_ptr<IPreProcessData>& data);

#define OV_PREPROC_PLUGIN_CALL_STATEMENT(...)                                                      \
    if (!_ptr)                                                                                     \
        IE_THROW() << "Wrapper used in the OV_PREPROC_PLUGIN_CALL_STATEMENT was not initialized."; \
    try {                                                                                          \
        __VA_ARGS__;                                                                               \
    } catch (...) {                                                                                \
        ::InferenceEngine::details::Rethrow();                                                     \
    }

class PreProcessDataPlugin {
    std::shared_ptr<void> _so = nullptr;
    std::shared_ptr<IPreProcessData> _ptr = nullptr;

public:
    PreProcessDataPlugin() {
#ifdef OPENVINO_STATIC_LIBRARY
#    ifdef ENABLE_GAPI_PREPROCESSING
        CreatePreProcessData(_ptr);
        if (!_ptr)
            IE_THROW() << "Failed to create IPreProcessData for G-API based preprocessing";
#   else
        IE_THROW() << "OpenVINO Runtime is compiled without G-API preprocessing support.\n"
                      "Use 'cmake -DENABLE_GAPI_PREPROCESSING=ON ...'";
#   endif // ENABLE_GAPI_PREPROCESSING
#else
        // preprocessing plugin can be found in the following locations
        // 1. in openvino-X.Y.Z location relative to libopenvino.so
        // 2. in the same folder as libopenvino.so

        ov::util::FilePath ovLibraryPath = getInferenceEngineLibraryPath();
        ov::util::FilePath libraryName = ov::util::to_file_path(std::string("openvino_gapi_preproc") + std::string(OV_BUILD_POSTFIX));
        libraryName = FileUtils::makePluginLibraryName({}, libraryName);

        std::ostringstream str;
        str << "openvino-" << IE_VERSION_MAJOR << "." << IE_VERSION_MINOR << "." << IE_VERSION_PATCH;
        ov::util::FilePath ovLibraryPathPlusOV = FileUtils::makePath(ovLibraryPath, ov::util::to_file_path(str.str()));

        const ov::util::FilePath preprocLibraryPathPlusOV = FileUtils::makePath(ovLibraryPathPlusOV, libraryName);
        const ov::util::FilePath preprocLibraryPath = FileUtils::makePath(ovLibraryPath, libraryName);
        const bool existsInOV = FileUtils::fileExist(preprocLibraryPathPlusOV);
        const bool existsInLib = FileUtils::fileExist(preprocLibraryPath);

        if (!existsInOV && !existsInLib) {
            IE_THROW() << "Please, make sure that pre-processing library "
                << ov::util::from_file_path(libraryName) << " is in "
                << ov::util::from_file_path(preprocLibraryPathPlusOV) << " or " << ov::util::from_file_path(preprocLibraryPath);
        }

        using CreateF = void(std::shared_ptr<IPreProcessData>& data);
        _so = ov::util::load_shared_object(existsInOV ? preprocLibraryPathPlusOV.c_str() : preprocLibraryPath.c_str());
        reinterpret_cast<CreateF *>(ov::util::get_symbol(_so, "CreatePreProcessData"))(_ptr);
#endif
    }

    void setRoiBlob(const Blob::Ptr &blob) {
        OV_PREPROC_PLUGIN_CALL_STATEMENT(_ptr->setRoiBlob(blob));
    }

    Blob::Ptr getRoiBlob() const {
        OV_PREPROC_PLUGIN_CALL_STATEMENT(return _ptr->getRoiBlob());
    }

    void execute(Blob::Ptr &preprocessedBlob, const PreProcessInfo& info, bool serial, int batchSize = -1) {
        OV_PREPROC_PLUGIN_CALL_STATEMENT(_ptr->execute(preprocessedBlob, info, serial, batchSize));
    }

    void isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst) {
        OV_PREPROC_PLUGIN_CALL_STATEMENT(_ptr->isApplicable(src, dst));
    }
};

#undef OV_PREPROC_PLUGIN_CALL_STATEMENT

using PreProcessDataPtr = std::shared_ptr<PreProcessDataPlugin>;

inline PreProcessDataPtr CreatePreprocDataHelper() {
    return std::make_shared<PreProcessDataPlugin>();
}

IE_SUPPRESS_DEPRECATED_END
}  // namespace InferenceEngine
