// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <memory>

#include <ie_blob.h>
#include <ie_profiling.hpp>
#include <ie_util_internal.hpp>
#include <file_utils.h>

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
class IPreProcessData : public details::IRelease {
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
    virtual void execute(Blob::Ptr &outBlob, const PreProcessInfo& info, bool serial, int batchSize = -1) = 0;

    virtual void isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst) = 0;
};

INFERENCE_PRERPOC_PLUGIN_API(StatusCode) CreatePreProcessData(IPreProcessData *& data, ResponseDesc *resp) noexcept;

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
    FileUtils::FilePath libraryName = FileUtils::toFilePath(std::string("inference_engine_preproc") + std::string("custom"));
    FileUtils::FilePath preprocLibraryPath = FileUtils::makeSharedLibraryName(getInferenceEngineLibraryPath(), libraryName);

    if (!FileUtils::fileExist(preprocLibraryPath)) {
        THROW_IE_EXCEPTION << "Please, make sure that pre-processing library "
            << FileUtils::fromFilePath(::FileUtils::makeSharedLibraryName({}, libraryName)) << " is in "
            << getIELibraryPath();
    }
    return PreProcessDataPtr(preprocLibraryPath);
}

//----------------------------------------------------------------------
//
// Implementation-internal types and functions and macros
//
//----------------------------------------------------------------------

namespace Resize {

static inline uint8_t saturateU32toU8(uint32_t v) {
    return static_cast<uint8_t>(v > UINT8_MAX ? UINT8_MAX : v);
}

void resize_bilinear_u8(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer);

void resize_area_u8_downscale(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer);

int getResizeAreaTabSize(int dst_go, int ssize, int dsize, float scale);

void computeResizeAreaTab(int src_go, int dst_go, int ssize, int dsize, float scale,
                          uint16_t* si, uint16_t* alpha, int max_count);

void generate_alpha_and_id_arrays(int x_max_count, int dcols, const uint16_t* xalpha, uint16_t* xsi,
                                  uint16_t** alpha, uint16_t** sxid);

enum BorderType {
    BORDER_CONSTANT  =  0,
    BORDER_REPLICATE =  1,
};

struct Border {
    BorderType  type;
    int32_t     value;
};

}  // namespace Resize

//----------------------------------------------------------------------

}  // namespace InferenceEngine
