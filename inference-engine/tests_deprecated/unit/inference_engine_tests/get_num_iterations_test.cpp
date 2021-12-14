// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/ie_layers.h>
#include <legacy/ie_layers_internal.hpp>
#include "single_layer_common.hpp"

namespace {

using InferenceEngine::details::operator<<;

struct From {
    explicit From(int new_value) : value(new_value) {}
    int value;
};

struct Axis {
    explicit Axis(int new_value) : value(new_value) {}
    int value;
};

struct Start {
    explicit Start(int new_value) : value(new_value) {}
    int value;
};

struct Stride {
    explicit Stride(int new_value) : value(new_value) {}
    int value;
};

struct End {
    explicit End(int new_value) : value(new_value) {}
    int value;
};

class PortMap : public InferenceEngine::TensorIterator::PortMap {
public:
    PortMap() = default;
    PortMap(From new_from, Axis new_axis, Start new_start, Stride new_stride, End new_end) : InferenceEngine::TensorIterator::PortMap() {
        from = new_from.value;
        to   = -1;
        axis   = new_axis.value;

        start  = new_start.value;
        stride = new_stride.value;
        end    = new_end.value;

        part_size = -1;
    }

    friend std::ostream& operator<<(std::ostream& stream, const PortMap& object) {
        stream << "{";
        stream << "from:" << object.from;
        stream << ", axis: " << object.axis;
        stream << ", start: " << object.start;
        stream << ", stride: " << object.stride;
        stream << ", end: " << object.end;
        stream << "}";
        return stream;;
    }
};

struct NegativeTestParams {
    std::vector<InferenceEngine::SizeVector> inputsDimensions;
    std::vector<PortMap> inputRules;

    std::vector<InferenceEngine::SizeVector> outputsDimensions;
    std::vector<PortMap> outputRules;

    friend std::ostream& operator<<(std::ostream& stream, const NegativeTestParams& object) {
        stream << "{";
        stream << "inputs: " << object.inputsDimensions;
        stream << ", inputRules: " << object.inputRules;
        stream << ", outputs: " << object.outputsDimensions;
        stream << ", outputRules: " << object.outputRules;
        stream << "}";
        return stream;
    }
};

class GetNumIterationsTests : public testing::Test {
public:
    void SetUp() override {
        tensorIterator = std::make_shared<InferenceEngine::TensorIterator>(
            InferenceEngine::LayerParams{"TI", "TensorIterator", InferenceEngine::Precision::UNSPECIFIED});
    }

    void TearDown() override {
        tensorIterator.reset();
    }

    void BuildTensorIterator(
        const std::vector<InferenceEngine::SizeVector>& inputsDimensions,
        const std::vector<PortMap>& inputRules,
        const std::vector<InferenceEngine::SizeVector>& outputsDimensions,
        const std::vector<PortMap>& outputRules) {
        std::transform(inputsDimensions.begin(),  inputsDimensions.end(),  std::back_inserter(inputsHandle), createData);
        std::transform(outputsDimensions.begin(), outputsDimensions.end(), std::back_inserter(tensorIterator->outData), createData);
        tensorIterator->insData = getInputs();

        tensorIterator->input_port_map  = getRules(inputRules);
        tensorIterator->output_port_map = getRules(outputRules);
    }

    int Run() const {
        return getNumIteration(*tensorIterator);
    }
private:
    static InferenceEngine::DataPtr createData(const InferenceEngine::SizeVector& dimensions) {
        const auto tensorDescriptor = InferenceEngine::TensorDesc{InferenceEngine::Precision::UNSPECIFIED, dimensions, InferenceEngine::Layout::ANY};
        return std::make_shared<InferenceEngine::Data>("", tensorDescriptor);
    }

    std::vector<InferenceEngine::DataWeakPtr> getInputs() const {
        std::vector<InferenceEngine::DataWeakPtr> inputs;
        std::transform(inputsHandle.begin(), inputsHandle.end(), std::back_inserter(inputs),
            [](const InferenceEngine::DataPtr& data) -> InferenceEngine::DataWeakPtr { return data; });
        return inputs;
    }

    static std::vector<InferenceEngine::TensorIterator::PortMap> getRules(const std::vector<PortMap>& testParam) {
        std::vector<InferenceEngine::TensorIterator::PortMap> rules;
        std::transform(testParam.begin(), testParam.end(), std::back_inserter(rules),
            [](const PortMap& rule) -> InferenceEngine::TensorIterator::PortMap { return rule; });
        return rules;
    }

    std::shared_ptr<InferenceEngine::TensorIterator> tensorIterator;
    std::vector<InferenceEngine::DataPtr> inputsHandle;
};

class GetNumIterationsNegativeTests : public GetNumIterationsTests, public testing::WithParamInterface<NegativeTestParams> {
public:
    void SetUp() override {
        GetNumIterationsTests::SetUp();

        const auto& parameters = GetParam();
        BuildTensorIterator(parameters.inputsDimensions, parameters.inputRules, parameters.outputsDimensions, parameters.outputRules);
    }
};

using GetNumIterationsInvalidAxisTests = GetNumIterationsNegativeTests;
TEST_P(GetNumIterationsInvalidAxisTests, InvalidAxisThrowsAnException) {
    ASSERT_ANY_THROW(Run());
}
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsInvalidAxisTests, testing::Values<NegativeTestParams>(
    NegativeTestParams{{{1}}, {{From{0}, Axis{-2}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}}, {{From{0}, Axis{10}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1, 2}}, {{From{0}, Axis{-2}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1, 2}}, {{From{0}, Axis{10}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1, 2}}, {{From{0}, Axis{1}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1, 2}, {1}}, {{From{1}, Axis{-2}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1, 2}, {1}}, {{From{1}, Axis{10}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1, 2}, {1}}, {{From{1}, Axis{1}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1}}, {{From{0}, Axis{-2}, Start{0}, Stride{1}, End{1}}, {From{1}, Axis{-2}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1}}, {{From{0}, Axis{10}, Start{0}, Stride{1}, End{1}}, {From{1}, Axis{10}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1}}, {{From{0}, Axis{1}, Start{0}, Stride{1}, End{1}}, {From{1}, Axis{1}, Start{0}, Stride{1}, End{1}}}, {}, {}},

    NegativeTestParams{{}, {}, {{1}}, {{From{0}, Axis{-2}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}}, {{From{0}, Axis{10}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1, 2}}, {{From{0}, Axis{-2}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1, 2}}, {{From{0}, Axis{10}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1, 2}}, {{From{0}, Axis{1}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1, 2}, {1}}, {{From{1}, Axis{-2}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1, 2}, {1}}, {{From{1}, Axis{10}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1, 2}, {1}}, {{From{1}, Axis{1}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1}}, {{From{0}, Axis{-2}, Start{0}, Stride{1}, End{1}}, {From{1}, Axis{-2}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1}}, {{From{0}, Axis{10}, Start{0}, Stride{1}, End{1}}, {From{1}, Axis{10}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1}}, {{From{0}, Axis{1}, Start{0}, Stride{1}, End{1}}, {From{1}, Axis{1}, Start{0}, Stride{1}, End{1}}}}
));

using GetNumIterationsInvalidStartTests = GetNumIterationsNegativeTests;
TEST_P(GetNumIterationsInvalidStartTests, InvalidStartThrowsAnException) {
    ASSERT_ANY_THROW(Run());
}
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsInvalidStartTests, testing::Values<NegativeTestParams>(
    NegativeTestParams{{{1}}, {{From{0}, Axis{0}, Start{2}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1, 2}}, {{From{0}, Axis{0}, Start{2}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1, 2}}, {{From{1}, Axis{0}, Start{2}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}, {1, 2}}, {{From{1}, Axis{1}, Start{2}, Stride{1}, End{1}}}, {}, {}},

    NegativeTestParams{{}, {}, {{1}}, {{From{0}, Axis{0}, Start{2}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1, 2}}, {{From{0}, Axis{0}, Start{2}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1, 2}}, {{From{1}, Axis{0}, Start{2}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}, {1, 2}}, {{From{1}, Axis{1}, Start{2}, Stride{1}, End{1}}}}
));

using GetNumIterationsInvalidEndTests = GetNumIterationsNegativeTests;
TEST_P(GetNumIterationsInvalidEndTests, InvalidEndThrowsAnException) {
    ASSERT_ANY_THROW(Run());
}
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsInvalidEndTests, testing::Values<NegativeTestParams>(
    NegativeTestParams{{}, {}, {{1}}, {{From{0}, Axis{0}, Start{0}, Stride{1}, End{2}}}},
    NegativeTestParams{{}, {}, {{1}, {1, 2}}, {{From{0}, Axis{0}, Start{0}, Stride{1}, End{2}}}},
    NegativeTestParams{{}, {}, {{1}, {1, 2}}, {{From{1}, Axis{0}, Start{0}, Stride{1}, End{2}}}}
));

using GetNumIterationsInvalidStrideTests = GetNumIterationsNegativeTests;
TEST_P(GetNumIterationsInvalidStrideTests, InvalidStrideThrowsAnException) {
    ASSERT_ANY_THROW(Run());
}
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsInvalidStrideTests, testing::Values<NegativeTestParams>(
    NegativeTestParams{{{1}}, {{From{0}, Axis{0}, Start{0}, Stride{0}, End{1}}}, {}, {}},
    NegativeTestParams{{}, {}, {{1}}, {{From{0}, Axis{0}, Start{0}, Stride{0}, End{1}}}}
));

using GetNumIterationsInvalidFromTests = GetNumIterationsNegativeTests;
TEST_P(GetNumIterationsInvalidFromTests, InvalidFromThrowsAnException) {
    ASSERT_ANY_THROW(Run());
}
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsInvalidFromTests, testing::Values<NegativeTestParams>(
    NegativeTestParams{{{1}}, {{From{-1}, Axis{0}, Start{0}, Stride{1}, End{1}}}, {}, {}},
    NegativeTestParams{{{1}}, {{From{1}, Axis{0}, Start{0}, Stride{1}, End{1}}}, {}, {}},

    NegativeTestParams{{}, {}, {{1}}, {{From{-1}, Axis{0}, Start{0}, Stride{1}, End{1}}}},
    NegativeTestParams{{}, {}, {{1}}, {{From{1}, Axis{0}, Start{0}, Stride{1}, End{1}}}}
));

using GetNumIterationsInvalidDirectionTests = GetNumIterationsNegativeTests;
TEST_P(GetNumIterationsInvalidDirectionTests, InvalidDirectionThrowsAnException) {
    ASSERT_ANY_THROW(Run());
}
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsInvalidDirectionTests, testing::Values<NegativeTestParams>(
    NegativeTestParams{{{10}}, {{From{0}, Axis{0}, Start{8}, Stride{1}, End{2}}}, {}, {}},
    NegativeTestParams{{{10}}, {{From{0}, Axis{0}, Start{2}, Stride{-1}, End{8}}}, {}, {}},

    NegativeTestParams{{}, {}, {{10}}, {{From{0}, Axis{0}, Start{8}, Stride{1}, End{2}}}},
    NegativeTestParams{{}, {}, {{10}}, {{From{0}, Axis{0}, Start{2}, Stride{-1}, End{8}}}}
));

using GetNumIterationsInvalidStepTests = GetNumIterationsNegativeTests;
TEST_P(GetNumIterationsInvalidStepTests, InvalidStepThrowsAnException) {
    ASSERT_ANY_THROW(Run());
}
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsInvalidStepTests, testing::Values<NegativeTestParams>(
    NegativeTestParams{{{10}}, {{From{0}, Axis{0}, Start{2}, Stride{3}, End{6}}}, {}, {}},
    NegativeTestParams{{{10}}, {{From{0}, Axis{0}, Start{2}, Stride{8}, End{6}}}, {}, {}},
    NegativeTestParams{{{10}}, {{From{0}, Axis{0}, Start{6}, Stride{-3}, End{2}}}, {}, {}},
    NegativeTestParams{{{10}}, {{From{0}, Axis{0}, Start{6}, Stride{-8}, End{2}}}, {}, {}},

    NegativeTestParams{{}, {}, {{10}}, {{From{0}, Axis{0}, Start{2}, Stride{3}, End{6}}}},
    NegativeTestParams{{}, {}, {{10}}, {{From{0}, Axis{0}, Start{2}, Stride{8}, End{6}}}},
    NegativeTestParams{{}, {}, {{10}}, {{From{0}, Axis{0}, Start{6}, Stride{-3}, End{2}}}},
    NegativeTestParams{{}, {}, {{10}}, {{From{0}, Axis{0}, Start{6}, Stride{-8}, End{2}}}}
));

using GetNumIterationsInvalidIterationNumbersTests = GetNumIterationsNegativeTests;
TEST_P(GetNumIterationsInvalidIterationNumbersTests, InvalidInterationNumbersThrowAnException) {
    ASSERT_ANY_THROW(Run());
}
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsInvalidIterationNumbersTests, testing::Values<NegativeTestParams>(
    NegativeTestParams{
        {
            {1, 3, 24, 24},
            {1, 3, 24, 24}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{1}, End{-1}},
            {From{1}, Axis{2}, Start{0}, Stride{2}, End{-1}}
        }, {}, {}
    },
    NegativeTestParams{
        {
            {1, 3, 24, 24},
            {1, 3, 24, 24},
            {1, 3, 48, 10}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{1}, End{-1}},
            {From{1}, Axis{2}, Start{0}, Stride{2}, End{-1}},
            {From{2}, Axis{3}, Start{0}, Stride{2}, End{-1}}
        }, {}, {}
    },
    NegativeTestParams{
        {
            {1, 3, 24, 24},
            {1, 3, 48, 24},
            {1, 3, 48, 10}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{1}, End{-1}},
            {From{1}, Axis{2}, Start{0}, Stride{2}, End{-1}},
            {From{2}, Axis{3}, Start{0}, Stride{2}, End{-1}}
        }, {}, {}
    },
    NegativeTestParams{
        {
            {1, 3, 24, 24},
            {1, 3, 24, 24}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{24}, End{-1}},
            {From{1}, Axis{2}, Start{0}, Stride{12}, End{-1}}
        }, {}, {}
    },

    NegativeTestParams{
        {}, {},
        {
            {1, 3, 24, 24},
            {1, 3, 24, 24}},
        {
            {From{0}, Axis{2}, Start{0}, Stride{1}, End{-1}},
            {From{1}, Axis{2}, Start{0}, Stride{2}, End{-1}}
        }
    },
    NegativeTestParams{
        {}, {},
        {
            {1, 3, 24, 24},
            {1, 3, 24, 24},
            {1, 3, 48, 10}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{1}, End{-1}},
            {From{1}, Axis{2}, Start{0}, Stride{2}, End{-1}},
            {From{2}, Axis{3}, Start{0}, Stride{2}, End{-1}}
        }
    },
    NegativeTestParams{
        {}, {},
        {
            {1, 3, 24, 24},
            {1, 3, 48, 24},
            {1, 3, 48, 10}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{1}, End{-1}},
            {From{1}, Axis{2}, Start{0}, Stride{2}, End{-1}},
            {From{2}, Axis{3}, Start{0}, Stride{2}, End{-1}}
        }
    },
    NegativeTestParams{
        {}, {},
        {
            {1, 3, 24, 24},
            {1, 3, 24, 24}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{24}, End{-1}},
            {From{1}, Axis{2}, Start{0}, Stride{12}, End{-1}}
        },
    },
    NegativeTestParams{
        {
            {1, 3, 24, 24}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{1}, End{-1}}
        },
        {
            {1, 3, 24, 24}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{2}, End{-1}}
        }
    },
    NegativeTestParams{
        {
            {1, 3, 24, 24},
            {1, 3, 10, 48}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{ 1}, End{-1}},
            {From{1}, Axis{3}, Start{-1}, Stride{-2}, End{0}},
        },
        {
            {1, 3, 48, 24},
            {1, 3, 24, 48}
        },
        {
            {From{0}, Axis{2}, Start{0}, Stride{1}, End{-1}},
            {From{1}, Axis{3}, Start{0}, Stride{1}, End{-1}},
        }
    }
));

struct PositiveTestParams {
    std::vector<InferenceEngine::SizeVector> inputsDimensions;
    std::vector<PortMap> inputRules;

    std::vector<InferenceEngine::SizeVector> outputsDimensions;
    std::vector<PortMap> outputRules;
    int reference;

    friend std::ostream& operator<<(std::ostream& stream, const PositiveTestParams& object) {
        stream << "{";
        stream << "inputs: " << object.inputsDimensions;
        stream << ", inputRules: " << object.inputRules;
        stream << ", outputs: " << object.outputsDimensions;
        stream << ", outputRules: " << object.outputRules;
        stream << ", reference: " << object.reference;
        stream << "}";
        return stream;
    }
};

class GetNumIterationsPositiveTests : public GetNumIterationsTests, public testing::WithParamInterface<PositiveTestParams> {
public:
    void SetUp() override {
        GetNumIterationsTests::SetUp();

        const auto& parameters = GetParam();
        BuildTensorIterator(parameters.inputsDimensions, parameters.inputRules, parameters.outputsDimensions, parameters.outputRules);
        reference = parameters.reference;
    }

    void Check() const {
        ASSERT_EQ(GetNumIterationsTests::Run(), reference);
    }
private:
    int reference = -1;
};

TEST_P(GetNumIterationsPositiveTests, CanGetNumIterations) {
    Check();
}
std::vector<PositiveTestParams> g_positiveTestParameters = {
    {{{1}}, {{From{0}, Axis{0}, Start{ 0}, Stride{ 1}, End{ 1}}}, {}, {}, 1},
    {{{1}}, {{From{0}, Axis{0}, Start{ 0}, Stride{ 1}, End{-1}}}, {}, {}, 1},
    {{{1}}, {{From{0}, Axis{0}, Start{-2}, Stride{ 1}, End{ 1}}}, {}, {}, 1},
    {{{1}}, {{From{0}, Axis{0}, Start{-2}, Stride{ 1}, End{-1}}}, {}, {}, 1},
    {{{1}}, {{From{0}, Axis{0}, Start{ 1}, Stride{-1}, End{ 0}}}, {}, {}, 1},
    {{{1}}, {{From{0}, Axis{0}, Start{-1}, Stride{-1}, End{ 0}}}, {}, {}, 1},
    {{{1}}, {{From{0}, Axis{0}, Start{ 1}, Stride{-1}, End{-2}}}, {}, {}, 1},
    {{{1}}, {{From{0}, Axis{0}, Start{-1}, Stride{-1}, End{-2}}}, {}, {}, 1},
    {{{7}}, {{From{0}, Axis{0}, Start{ 0}, Stride{ 7}, End{ 7}}}, {}, {}, 1},
    {{{7}}, {{From{0}, Axis{0}, Start{-1}, Stride{-7}, End{ 0}}}, {}, {}, 1},
    {{{1, 1}}, {{From{0}, Axis{1}, Start{ 0}, Stride{ 1}, End{ 1}}}, {}, {}, 1},
    {{{1, 1}}, {{From{0}, Axis{1}, Start{ 0}, Stride{ 1}, End{-1}}}, {}, {}, 1},
    {{{1, 1}}, {{From{0}, Axis{1}, Start{-2}, Stride{ 1}, End{ 1}}}, {}, {}, 1},
    {{{1, 1}}, {{From{0}, Axis{1}, Start{-2}, Stride{ 1}, End{-1}}}, {}, {}, 1},
    {{{1, 1}}, {{From{0}, Axis{1}, Start{ 1}, Stride{-1}, End{ 0}}}, {}, {}, 1},
    {{{1, 1}}, {{From{0}, Axis{1}, Start{-1}, Stride{-1}, End{ 0}}}, {}, {}, 1},
    {{{1, 1}}, {{From{0}, Axis{1}, Start{ 1}, Stride{-1}, End{-2}}}, {}, {}, 1},
    {{{1, 1}}, {{From{0}, Axis{1}, Start{-1}, Stride{-1}, End{-2}}}, {}, {}, 1},
    {{{1, 7}}, {{From{0}, Axis{1}, Start{ 0}, Stride{ 7}, End{ 7}}}, {}, {}, 1},
    {{{1, 7}}, {{From{0}, Axis{1}, Start{-1}, Stride{-7}, End{ 0}}}, {}, {}, 1},
    {{{13}}, {{From{0}, Axis{0}, Start{  3}, Stride{ 1}, End{ 13}}}, {}, {}, 10},
    {{{13}}, {{From{0}, Axis{0}, Start{-11}, Stride{ 1}, End{ 13}}}, {}, {}, 10},
    {{{13}}, {{From{0}, Axis{0}, Start{  3}, Stride{ 2}, End{ 13}}}, {}, {}, 5},
    {{{13}}, {{From{0}, Axis{0}, Start{-11}, Stride{ 2}, End{ 13}}}, {}, {}, 5},
    {{{13}}, {{From{0}, Axis{0}, Start{  0}, Stride{ 1}, End{ 10}}}, {}, {}, 10},
    {{{13}}, {{From{0}, Axis{0}, Start{  0}, Stride{ 1}, End{ -4}}}, {}, {}, 10},
    {{{13}}, {{From{0}, Axis{0}, Start{  0}, Stride{ 2}, End{ 10}}}, {}, {}, 5},
    {{{13}}, {{From{0}, Axis{0}, Start{  0}, Stride{ 2}, End{ -4}}}, {}, {}, 5},
    {{{13}}, {{From{0}, Axis{0}, Start{  3}, Stride{ 1}, End{ 10}}}, {}, {}, 7},
    {{{13}}, {{From{0}, Axis{0}, Start{-11}, Stride{ 1}, End{ -4}}}, {}, {}, 7},
    {{{13}}, {{From{0}, Axis{0}, Start{  3}, Stride{ 2}, End{  9}}}, {}, {}, 3},
    {{{13}}, {{From{0}, Axis{0}, Start{-11}, Stride{ 2}, End{ -5}}}, {}, {}, 3},
    {{{13}}, {{From{0}, Axis{0}, Start{  3}, Stride{ 1}, End{ -4}}}, {}, {}, 7},
    {{{13}}, {{From{0}, Axis{0}, Start{  3}, Stride{ 2}, End{ -5}}}, {}, {}, 3},
    {{{13}}, {{From{0}, Axis{0}, Start{-11}, Stride{ 1}, End{ 10}}}, {}, {}, 7},
    {{{13}}, {{From{0}, Axis{0}, Start{-11}, Stride{ 2}, End{  9}}}, {}, {}, 3},
    {{{13}}, {{From{0}, Axis{0}, Start{ 13}, Stride{-1}, End{  3}}}, {}, {}, 10},
    {{{13}}, {{From{0}, Axis{0}, Start{ 13}, Stride{-1}, End{-11}}}, {}, {}, 10},
    {{{13}}, {{From{0}, Axis{0}, Start{ 13}, Stride{-2}, End{  3}}}, {}, {}, 5},
    {{{13}}, {{From{0}, Axis{0}, Start{ 13}, Stride{-2}, End{-11}}}, {}, {}, 5},
    {{{13}}, {{From{0}, Axis{0}, Start{ 10}, Stride{-1}, End{  0}}}, {}, {}, 10},
    {{{13}}, {{From{0}, Axis{0}, Start{ -4}, Stride{-1}, End{  0}}}, {}, {}, 10},
    {{{13}}, {{From{0}, Axis{0}, Start{ 10}, Stride{-2}, End{  0}}}, {}, {}, 5},
    {{{13}}, {{From{0}, Axis{0}, Start{ -4}, Stride{-2}, End{  0}}}, {}, {}, 5},
    {{{13}}, {{From{0}, Axis{0}, Start{ 10}, Stride{-1}, End{  3}}}, {}, {}, 7},
    {{{13}}, {{From{0}, Axis{0}, Start{ -4}, Stride{-1}, End{-11}}}, {}, {}, 7},
    {{{13}}, {{From{0}, Axis{0}, Start{  9}, Stride{-2}, End{  3}}}, {}, {}, 3},
    {{{13}}, {{From{0}, Axis{0}, Start{ -5}, Stride{-2}, End{-11}}}, {}, {}, 3},
    {{{13}}, {{From{0}, Axis{0}, Start{ -4}, Stride{-1}, End{  3}}}, {}, {}, 7},
    {{{13}}, {{From{0}, Axis{0}, Start{ -5}, Stride{-2}, End{  3}}}, {}, {}, 3},
    {{{13}}, {{From{0}, Axis{0}, Start{ 10}, Stride{-1}, End{-11}}}, {}, {}, 7},
    {{{13}}, {{From{0}, Axis{0}, Start{  9}, Stride{-2}, End{-11}}}, {}, {}, 3}
};
INSTANTIATE_TEST_SUITE_P(NumIterationsTest, GetNumIterationsPositiveTests, testing::ValuesIn(g_positiveTestParameters));

}
