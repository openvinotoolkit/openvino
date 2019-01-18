// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"
#include "ie_input_info.hpp"

#include <tuple>
#include <vector>
#include <opencv2/gapi/gcompiled.hpp>
#include <opencv2/gapi/util/optional.hpp>
#include "ie_profiling.hpp"

// FIXME: Move this definition back to ie_preprocess_data,
// also free ie_preprocess_gapi of these details

namespace InferenceEngine {

class PreprocEngine {
    using BlobDesc = std::tuple<Precision, Layout, SizeVector>;
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

public:
    PreprocEngine();
    bool preprocessWithGAPI(Blob::Ptr &inBlob, Blob::Ptr &outBlob, const ResizeAlgorithm &algorithm, bool omp_serial);
};

}  // namespace InferenceEngine
