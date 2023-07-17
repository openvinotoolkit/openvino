// Copyright (C) 2018-2023 Intel Corporation
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
#include <openvino/itt.hpp>

// FIXME: Move this definition back to ie_preprocess_data,
// also free ie_preprocess_gapi of these details

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START
class PreprocEngine {
    using BlobDesc = std::tuple<Precision, Layout, SizeVector, ColorFormat>;
    using CallDesc = std::tuple<BlobDesc, BlobDesc, ResizeAlgorithm>;
    template<typename T> using Opt = cv::util::optional<T>;

    Opt<CallDesc> _lastCall;
    std::vector<cv::GCompiled> _lastComp;

    openvino::itt::handle_t _perf_graph_building = openvino::itt::handle("Preproc Graph Building");
    openvino::itt::handle_t _perf_exec_tile = openvino::itt::handle("Preproc Calc Tile");
    openvino::itt::handle_t _perf_exec_graph = openvino::itt::handle("Preproc Exec Graph");
    openvino::itt::handle_t _perf_graph_compiling = openvino::itt::handle("Preproc Graph compiling");

    enum class Update { REBUILD, RESHAPE, NOTHING };
    Update needUpdate(const CallDesc &newCall) const;

    void executeGraph(Opt<cv::GComputation>& lastComputation,
                      const std::vector<std::vector<cv::gapi::own::Mat>>& src,
                      std::vector<std::vector<cv::gapi::own::Mat>>& dst,
                      int batch_size,
                      bool omp_serial,
                      Update update);

    template<typename BlobTypePtr>
    void preprocessBlob(const BlobTypePtr &inBlob, MemoryBlob::Ptr &outBlob,
        ResizeAlgorithm algorithm, ColorFormat in_fmt, ColorFormat out_fmt, bool omp_serial,
        int batch_size);

public:
    PreprocEngine();
    static void checkApplicabilityGAPI(const Blob::Ptr &src, const Blob::Ptr &dst);
    static int getCorrectBatchSize(int batch_size, const Blob::Ptr& roiBlob);
    void preprocessWithGAPI(const Blob::Ptr &inBlob, Blob::Ptr &outBlob, const ResizeAlgorithm &algorithm,
        ColorFormat in_fmt, bool omp_serial, int batch_size = -1);
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
