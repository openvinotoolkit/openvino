// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <memory>

#include "ie_blob.h"
#include "ie_input_info.hpp"
#include "ie_profiling.hpp"

namespace InferenceEngine {

class PreprocEngine;

/**
 * @brief This class stores pre-process information for exact input
 */
class INFERENCE_ENGINE_API_CLASS(PreProcessData) {
    /**
     * @brief ROI blob.
     */
    Blob::Ptr _roiBlob = nullptr;
    Blob::Ptr _tmp1 = nullptr;
    Blob::Ptr _tmp2 = nullptr;

    /**
     * @brief Pointer-to-implementation (PIMPL) hiding preprocessing implementation details.
     * BEWARE! Will be shared among copies!
     */
    std::shared_ptr<PreprocEngine> _preproc;

    InferenceEngine::ProfilingTask perf_resize {"Resize"};
    InferenceEngine::ProfilingTask perf_reorder_before {"Reorder before"};
    InferenceEngine::ProfilingTask perf_reorder_after {"Reorder after"};
    InferenceEngine::ProfilingTask perf_preprocessing {"Preprocessing"};

public:
    /**
     * @brief Sets ROI blob to be resized and placed to the default input blob during pre-processing.
     * @param blob ROI blob.
     */
    void setRoiBlob(const Blob::Ptr &blob);

    /**
     * @brief Gets pointer to the ROI blob used for a given input.
     * @return Blob pointer.
     */
    Blob::Ptr getRoiBlob() const;

    /**
     * @brief Executes input pre-processing with a given pre-processing information.
     * @param outBlob pre-processed output blob to be used for inference.
     * @param info pre-processing info that specifies resize algorithm and color format.
     * @param serial disable OpenMP threading if the value set to true.
     * @param batchSize batch size for pre-processing.
     */
    void execute(Blob::Ptr &outBlob, const PreProcessInfo& info, bool serial, int batchSize = -1);

    static void isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst);
};

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
