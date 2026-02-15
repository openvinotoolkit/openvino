//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scenario/inference.hpp"

#include <memory>

struct InOutLayers {
    LayersInfo in_layers;
    LayersInfo out_layers;
};

class OpenVINOLayersReader {
public:
    OpenVINOLayersReader();
    InOutLayers readLayers(const OpenVINOParams& params, const bool use_results_names = false);

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

namespace LayersReader {
InOutLayers readLayers(const InferenceParams& params);
}  // namespace LayersReader
