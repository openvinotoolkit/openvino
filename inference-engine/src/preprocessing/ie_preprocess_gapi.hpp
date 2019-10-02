// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"
#include "ie_compound_blob.h"
#include "ie_input_info.hpp"

#include <tuple>
#include <vector>
#include <opencv2/gapi/gcompiled.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/util/optional.hpp>
#include "ie_profiling.hpp"

// FIXME: Move this definition back to ie_preprocess_data,
// also free ie_preprocess_gapi of these details

namespace InferenceEngine {

class PreprocEngine {
    using BlobDesc = std::tuple<Precision, Layout, SizeVector, ColorFormat>;
    using CallDesc = std::tuple<BlobDesc, BlobDesc, ResizeAlgorithm>;
    template<typename T> using Opt = cv::util::optional<T>;

    Opt<CallDesc> _lastCall;
    std::vector<cv::GCompiled> _lastComp;

    ProfilingTask _perf_graph_building {"Preproc Graph Building"};
    ProfilingTask _perf_exec_tile  {"Preproc Calc Tile"};
    ProfilingTask _perf_exec_graph {"Preproc Exec Graph"};
    ProfilingTask _perf_graph_compiling {"Preproc Graph compiling"};

    enum class Update { REBUILD, RESHAPE, NOTHING };
    Update needUpdate(const CallDesc &newCall) const;

    void executeGraph(Opt<cv::GComputation>& lastComputation,
                      const std::vector<std::vector<cv::gapi::own::Mat>>& src,
                      std::vector<std::vector<cv::gapi::own::Mat>>& dst,
                      int batch_size,
                      bool omp_serial,
                      Update update);

    bool preprocessBlob(const MemoryBlob::Ptr &inBlob, MemoryBlob::Ptr &outBlob,
        ResizeAlgorithm algorithm, ColorFormat in_fmt, ColorFormat out_fmt, bool omp_serial,
        int batch_size);

    bool preprocessBlob(const NV12Blob::Ptr &inBlob, MemoryBlob::Ptr &outBlob,
        ResizeAlgorithm algorithm, ColorFormat in_fmt, ColorFormat out_fmt, bool omp_serial,
        int batch_size);

public:
    PreprocEngine();
    static bool useGAPI();
    static void checkApplicabilityGAPI(const Blob::Ptr &src, const Blob::Ptr &dst);
    static int getCorrectBatchSize(int batch_size, const Blob::Ptr& roiBlob);
    bool preprocessWithGAPI(Blob::Ptr &inBlob, Blob::Ptr &outBlob, const ResizeAlgorithm &algorithm,
        ColorFormat in_fmt, bool omp_serial, int batch_size = -1);
};

}  // namespace InferenceEngine
