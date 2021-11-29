// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <memory>

#include <ie_blob.h>
#include <file_utils.h>
#include <ie_preprocess.hpp>

#include <details/ie_so_pointer.hpp>

namespace InferenceEngine {

#if defined(_WIN32)
    #ifdef IMPLEMENT_PREPROC_PLUGIN
        #define INFERENCE_PRERPOC_PLUGIN_API(type) extern "C"   __declspec(dllexport) type
    #else
        #define INFERENCE_PRERPOC_PLUGIN_API(type) extern "C" type
    #endif
#elif(__GNUC__ >= 4)
    #define INFERENCE_PRERPOC_PLUGIN_API(type) extern "C"   __attribute__((visibility("default"))) type
#else
    #define INFERENCE_PRERPOC_PLUGIN_API(TYPE) extern "C" TYPE
#endif

/**
 * @brief This class stores pre-process information for exact input
 */
class IPreProcessData : public std::enable_shared_from_this<IPreProcessData> {
public:
    /**
     * @brief Sets ROI blob to be resized and placed to the default input blob during pre-processing.
     * @param blob ROI blob.
     */
    //FIXME: rename to setUserBlob
    virtual void setRoiBlob(const Blob::Ptr &blob) = 0;

    /**
     * @brief Gets pointer to the ROI blob used for a given input.
     * @return Blob pointer.
     */
    //FIXME: rename to getUserBlob
    virtual Blob::Ptr getRoiBlob() const = 0;

    /**
     * @brief Executes input pre-processing with a given pre-processing information.
     * @param outBlob pre-processed output blob to be used for inference.
     * @param info pre-processing info that specifies resize algorithm and color format.
     * @param serial disable OpenMP threading if the value set to true.
     * @param batchSize batch size for pre-processing.
     */
    virtual void execute(Blob::Ptr &preprocessedBlob, const PreProcessInfo& info, bool serial, int batchSize = -1) = 0;

    //FIXME: rename to verifyAplicable
    virtual void isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst) = 0;

protected:
    ~IPreProcessData() = default;
};

INFERENCE_PRERPOC_PLUGIN_API(void) CreatePreProcessData(std::shared_ptr<IPreProcessData>& data);

namespace details {

/**
 * @brief This class defines the name of the fabric for creating an IHeteroInferencePlugin object in DLL
 */
template<>
class SOCreatorTrait<IPreProcessData> {
public:
    /**
     * @brief A name of the fabric for creating IInferencePlugin object in DLL
     */
    static constexpr auto name = "CreatePreProcessData";
};

}  // namespace details

/**
 * @brief A C++ helper to work with objects created by the plugin.
 * Implements different interfaces.
 */
using PreProcessDataPtr = InferenceEngine::details::SOPointer<IPreProcessData>;

inline PreProcessDataPtr CreatePreprocDataHelper() {
    FileUtils::FilePath libraryName = FileUtils::toFilePath(std::string("inference_engine_preproc") + std::string(IE_BUILD_POSTFIX));
    FileUtils::FilePath preprocLibraryPath = FileUtils::makePluginLibraryName(getInferenceEngineLibraryPath(), libraryName);

    if (!FileUtils::fileExist(preprocLibraryPath)) {
        IE_THROW() << "Please, make sure that pre-processing library "
            << FileUtils::fromFilePath(::FileUtils::makePluginLibraryName({}, libraryName)) << " is in "
            << getIELibraryPath();
    }
    return {preprocLibraryPath};
}

}  // namespace InferenceEngine
