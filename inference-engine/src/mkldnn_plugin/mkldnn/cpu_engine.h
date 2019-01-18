// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include "desc_layer.h"
#include "desc_tensor.h"
#include "desc_tensor_comb.h"

#include "cpu_prim_layer.h"
#include "cpu_prim_tensor.h"

#include "mkldnn.hpp"
#include <memory>
#include <vector>

using namespace InferenceEngine;

namespace MKLDNNPlugin {
class CpuEngine;

using CpuEnginePtr = std::shared_ptr<CpuEngine>;

class CpuEngine : public details::no_copy {
public:
    CpuEngine() : eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0)) {}

    void bindThreads();

    void createDescription(DescTensorPtr tns, bool isWeights = false);

    void createDescription(DescLayerPtr layer);

    void setFlatFormat(DescTensorPtr tns);

    void createPrimitive(DescTensorPtr tns);

    void createPrimitive(DescLayerPtr tns);

    void setData(const TBlob<float> &src, DescTensorPtr dst);

    void getData(const DescTensorPtr src, TBlob<float> &dst);

    void subtraction(DescTensorPtr dst, DescTensorPtr sub);

    void subtraction(DescTensorPtr dst, std::vector<float> sub);

    void score(std::vector<DescLayerPtr> layers);

    void score(DescLayerPtr layer);

    void process(std::vector<mkldnn::primitive> exec_queue);

    mkldnn::engine eng;  // TODO: Move me back to private section

private:
    static inline mkldnn::memory::desc *get_desc(std::vector<DescTensorPtr> tensors, size_t indx = 0);

    static inline mkldnn::memory::desc *get_desc(DescTensorPtr tns);

    static inline mkldnn::memory *get_prim(std::vector<DescTensorPtr> tns, size_t indx = 0);

    static inline mkldnn::memory *get_prim(DescTensorPtr tns);

    void createPrimitiveCombined(DescTensorComb &tns, void *data);
};
}  // namespace MKLDNNPlugin