// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <ie_layers.h>
#include "precision_utils.h"
#include "ie_parallel.hpp"

using namespace InferenceEngine::PrecisionUtils;

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for TBD layer
 */
class RandomUniformConstInfer : public ConstInferImpl {
public:
    explicit RandomUniformConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
	// extract params from IR model, default values from
	// tensorflow.python.ops.random_ops
	// tensorflow.python.framework.random_seed
	int seed = 0, seed2 = 0x7FFFffff;
	if (params.find("seed") != params.end()) {
	    seed = std::stoi(params.find("seed")->second);
	}
        if (params.find("seed2") != params.end()) {
            seed2 = std::stoi(params.find("seed2")->second);
        }
	if (seed == 0 && seed2 == 0) {
	    seed2 = 0x7FFFffff;
	}
	std::seed_seq s = {seed, seed2};
        std::default_random_engine random(s);
        float minval = 0.0, maxval = 1.0;
        if (params.find("minval") != params.end()) {
            minval = std::stof(params.find("minval")->second);
        }
        if (params.find("maxval") != params.end()) {
            maxval = std::stof(params.find("maxval")->second);
        }
        std::uniform_real_distribution<float> dis(minval, maxval);

	// execute
	auto outBlob = *outData.begin();
        if (outBlob->getTensorDesc().getPrecision() == Precision::FP16) {
            const auto* inBuffer = inData[0]->cbuffer().as<ie_fp16*>();
            auto* outBuffer = outData[0]->buffer().as<ie_fp16*>();
            parallel_for(outBlob->size(), [&](size_t i) {
                outBuffer[i] = f32tof16(dis(random));
            });
        } else {
            const auto* inBuffer = inData[0]->cbuffer().as<float*>();
            auto* outBuffer = outData[0]->buffer().as<float*>();
            parallel_for(outBlob->size(), [&](size_t i) {
                outBuffer[i] = dis(random);
            });
	}
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
