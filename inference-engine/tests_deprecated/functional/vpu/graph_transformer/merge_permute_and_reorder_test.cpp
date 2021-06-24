// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gt_functional_tests.hpp"

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/utils/logger.hpp>

using namespace InferenceEngine;
using namespace vpu;

struct PermutationStage {
    PermutationIndexVector permute;
    const DimsOrder * reorder = nullptr; // use pointer to avoid copy of uninitialized global static object.
    PermutationStage() = default;
    PermutationStage(const PermutationIndexVector& permute) : permute(permute) {}
    PermutationStage(const DimsOrder& reorder) : reorder(&reorder) {}
};

static inline void PrintTo(const PermutationStage& param, ::std::ostream* os) {
    if (!param.permute.empty())
        *os << ::testing::PrintToString(param.permute);
    else
        printTo(*os, *param.reorder);
}

using PermutationsSequence = std::vector<PermutationStage>;

using MergePermuteNDParams = std::tuple<InferenceEngine::SizeVector,  // input tensor sizes
                                        PermutationsSequence>;        // permutation vectors sequence

class myriadGTMergePermuteNDTests_nightly:
    public graphTransformerFunctionalTests,
    public testing::WithParamInterface<MergePermuteNDParams> {

    using PermuteDims = PermutationDimsMap;
    static constexpr DataType defaultDataType = DataType::FP16;
protected:
    DimValues MakeStubDimValues(const DimsOrder& layout, const SizeVector& dims) {
        const auto perm = layout.toPermutation();
        const int numDms = layout.numDims();
        DimValues dimValues;
        for (int ind = 0; ind < numDms; ++ind) {
            dimValues.set(perm[ind], static_cast<int>(dims[ind]));
        }
        return dimValues;
    }

    InferenceEngine::SizeVector applyPermute(const InferenceEngine::SizeVector& dims, const PermutationIndexVector& permute) {
        InferenceEngine::SizeVector result(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            result[i] = dims[permute[i]];
        }
        return result;
    }

    int64_t InferPermuteSequence(InferenceEngine::SizeVector inputTensorSizes,
                                 const PermutationsSequence& permutationVectors,
                                 const bool usePermuteMerging,
                                 Blob::Ptr& outputBlob) {
        PrepareGraphCompilation();
        _configuration.compileConfig().detectBatch = false;
        _configuration.compileConfig().enablePermuteMerging = usePermuteMerging;

        IE_ASSERT(permutationVectors.size() >= 2);

        DimsOrder layout = *permutationVectors[0].reorder; // first "reorder" is fake and determines input layout.

        InitializeInputData({defaultDataType, layout, MakeStubDimValues(layout, inputTensorSizes)});

        for (int i = 1; i < permutationVectors.size(); ++i) {
            const bool lastIteration = i == permutationVectors.size() - 1;
            const auto permutationStep = permutationVectors[i];
            PermutationIndexVector permute = permutationStep.permute;
            if (permutationStep.permute.empty()) {
                auto oldLayout = layout;
                layout = *permutationStep.reorder;
                permute = calculatePermuteForReorder(oldLayout, layout);
            }

            inputTensorSizes = applyPermute(inputTensorSizes, permute);

            const DataDesc intermediateDataDesc(defaultDataType, layout,  MakeStubDimValues(layout, inputTensorSizes));

            vpu::Data dataInt = lastIteration
                              ? InitializeOutputData(intermediateDataDesc)
                              : _gtModel->addNewData("data" + std::to_string(i), intermediateDataDesc);
            if (!permutationStep.permute.empty()) {
                _stageBuilder->addPermuteStage(_gtModel,
                                               "Permute" + std::to_string(i),
                                               nullptr,
                                               _dataIntermediate,
                                               dataInt,
                                               permuteVectorToMap(permutationStep.permute, layout, layout)
                                               );
            } else {
                _stageBuilder->addReorderStage(_gtModel,
                                               "Reorder" + std::to_string(i),
                                               nullptr,
                                               _dataIntermediate,
                                               dataInt
                                               );
            }
            _dataIntermediate = dataInt;
        }

        Blob::Ptr inputBlob;
        return CompileAndInfer(inputBlob, outputBlob);
    }
};

static const std::vector<InferenceEngine::SizeVector> s_inTensors_3D = {
    {5, 7, 11},
};

static const std::vector<PermutationsSequence> s_permuteParams_3D = {
    {DimsOrder::CHW, PermutationIndexVector{0, 1, 2}},
    {DimsOrder::CHW, PermutationIndexVector{1, 2, 0}, DimsOrder::HWC},
    {DimsOrder::CHW, PermutationIndexVector{1, 2, 0}, DimsOrder::HWC},
    {DimsOrder::CHW, PermutationIndexVector{1, 2, 0}, DimsOrder::HWC, PermutationIndexVector{1, 2, 0}, DimsOrder::HCW},
};

static const std::vector<InferenceEngine::SizeVector> s_inTensors_5D = {
    {2, 3, 5, 7, 11},
};

static const std::vector<PermutationsSequence> s_permuteParams_5D = {
    {DimsOrder::NCDHW, PermutationIndexVector{0, 1, 2, 3, 4}, DimsOrder::NDHWC},
    {DimsOrder::NDHWC, PermutationIndexVector{1, 2, 3, 4, 0}, DimsOrder::NCDHW, PermutationIndexVector{1, 2, 3, 4, 0}, DimsOrder::NDHWC},
    {DimsOrder::NCDHW, DimsOrder::NDHWC, DimsOrder::NCDHW},
};

TEST_P(myriadGTMergePermuteNDTests_nightly, Permute) {
    const auto& test_params = GetParam();
    const auto& inputTensorSizes   = std::get<0>(test_params);
    const auto& permutationVectors = std::get<1>(test_params);

    Blob::Ptr outputBlobWithMerging, outputBlobWithoutMerging;

    const auto executionMicrosecondsOptimized = InferPermuteSequence(inputTensorSizes, permutationVectors, true  , outputBlobWithMerging);
    const auto executionMicroseconds          = InferPermuteSequence(inputTensorSizes, permutationVectors, false , outputBlobWithoutMerging);

    CompareCommonAbsolute(outputBlobWithMerging, outputBlobWithoutMerging, 0.);
    std::cout << "Myriad time = non-optimized: " << executionMicroseconds << " us., optimized: " << executionMicrosecondsOptimized << " us.\n";
}

INSTANTIATE_TEST_SUITE_P(accuracy_3D, myriadGTMergePermuteNDTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_3D)
          , ::testing::ValuesIn(s_permuteParams_3D)
));

INSTANTIATE_TEST_SUITE_P(accuracy_5D, myriadGTMergePermuteNDTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors_5D)
          , ::testing::ValuesIn(s_permuteParams_5D)
));
