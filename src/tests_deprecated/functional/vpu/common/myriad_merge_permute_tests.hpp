// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#include <debug.h>

using namespace InferenceEngine;

using PermutationsSequence = std::vector<InferenceEngine::SizeVector>;

using MergePermuteNDParams = std::tuple<InferenceEngine::SizeVector,  // input tensor sizes
                                        PermutationsSequence>;        // permutation vectors sequence

class myriadLayersMergePermuteNDTests_nightly:
    public myriadLayersTests_nightly,
    public testing::WithParamInterface<MergePermuteNDParams> {

public:
    Blob::Ptr InferPermute(InferenceEngine::SizeVector input_tensor_sizes,
                           const PermutationsSequence& permutation_vectors,
                           const bool usePermuteMerging,
                           int64_t& executionMicroseconds)
    {
        ResetGeneratedNet();
        ResetReferenceLayers();

        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH  ] = CONFIG_VALUE(NO);
        _config[InferenceEngine::MYRIAD_ENABLE_PERMUTE_MERGING] = usePermuteMerging ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO) ;

        for (const auto& permutation_vector : permutation_vectors) {
            const auto num_dims = input_tensor_sizes.size();
            SizeVector output_tensor_sizes(num_dims);
            for (size_t i = 0; i < num_dims; i++) {
                output_tensor_sizes[i] = input_tensor_sizes[permutation_vector[i]];
            }

            const std::map<std::string, std::string> layer_params{{"order", details::joinVec(permutation_vector)}};

            _testNet.addLayer(LayerInitParams("Permute")
                     .params(layer_params)
                     .in({input_tensor_sizes})
                     .out({output_tensor_sizes}),
                     ref_permute_wrap);

            input_tensor_sizes = output_tensor_sizes;  // update input for next layer
        }

        IE_ASSERT(generateNetAndInfer(NetworkInitParams().useHWOpt(CheckMyriadX()).runRefGraph(false)));

        auto perfMap = _inferRequest.GetPerformanceCounts();

        executionMicroseconds = 0;
        for (const auto& perfPair : perfMap) {
            const InferenceEngine::InferenceEngineProfileInfo& info = perfPair.second;
            if (info.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
                executionMicroseconds += info.realTime_uSec;
            }
        }

        return _outputMap.begin()->second;
    }
};

TEST_P(myriadLayersMergePermuteNDTests_nightly, Permute) {
    const auto& test_params = GetParam();
    const auto& input_tensor_sizes  = std::get<0>(test_params);
    const auto& permutation_vectors = std::get<1>(test_params);

    int64_t executionMicroseconds = 0, executionMicrosecondsOptimized = 0;

    const auto output_blob_with_merging    = InferPermute(input_tensor_sizes, permutation_vectors, true  , executionMicrosecondsOptimized);
    const auto output_blob_without_merging = InferPermute(input_tensor_sizes, permutation_vectors, false , executionMicroseconds);

    CompareCommonAbsolute(output_blob_with_merging, output_blob_without_merging, 0.);
    std::cout << "Myriad time = non-optimized: " << executionMicroseconds << " us., optimized: " << executionMicrosecondsOptimized << " us.\n";
}
static const std::vector<InferenceEngine::SizeVector> s_inTensors_3D = {
    {5, 7, 11},
    {1, 3, 4},
};

static const std::vector<PermutationsSequence> s_permuteParams_3D = {
    {{0, 1, 2}, {1, 0, 2}}, // trivial for case with dims {1, 3, 4}
    {{1, 2, 0}, {1, 2, 0}},
    {{1, 2, 0}, {1, 2, 0}, {1, 2, 0}}, // trivial one.
};

static const std::vector<InferenceEngine::SizeVector> s_inTensors_4D = {
    {3, 5, 7, 11},
    {5, 1, 1, 7},
};

static const std::vector<PermutationsSequence> s_permuteParams_4D = {
    {{0, 1, 2, 3}, {1, 0, 3, 2}}, // 
    {{0, 1, 2, 3}, {0, 1, 3, 2}}, // trivial for case with dims {5, 1, 1, 7}
    {{1, 2, 3, 0}, {1, 2, 3, 0}, {1, 2, 3, 0}},
    {{1, 2, 3, 0}, {1, 2, 3, 0}, {1, 2, 3, 0}, {1, 2, 3, 0}}, // trivial one.
};

static const std::vector<InferenceEngine::SizeVector> s_inTensors_5D = {
    {2, 3, 5, 7, 11},
    {2, 3, 1, 7, 11},
};
static const std::vector<PermutationsSequence> s_permuteParams_5D = {
    {{0, 1, 2, 3, 4}, {0, 1, 3, 2, 4}}, //
    {{0, 1, 2, 3, 4}, {0, 2, 1, 3, 4}}, // trivial for case with dims {2, 3, 1, 7, 11}
    {{0, 4, 1, 2, 3}, {0, 2, 1, 3, 4}},
    {{0, 3, 4, 1, 2}, {0, 1, 3, 2, 4}},
    {{1, 2, 3, 4, 0}, {1, 2, 3, 4, 0}},
    {{4, 0, 1, 2, 3}, {4, 0, 1, 2, 3}},
    {{0, 1, 2, 3, 4}, {0, 3, 1, 2, 4}, {0, 1, 2, 4, 3}},
    {{0, 1, 3, 2, 4}, {3, 1, 0, 2, 4}, {0, 1, 2, 4, 3}, {4, 1, 2, 0, 3}},
    {{1, 2, 3, 4, 0}, {1, 2, 3, 4, 0}, {1, 2, 3, 4, 0}, {1, 2, 3, 4, 0}, {1, 2, 3, 4, 0}}, // trivial one.
};
