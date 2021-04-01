// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reference_functions.hpp"
#include "myriad_layers_tests.hpp"
#include "tests_vpu_common.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <limits>

using namespace InferenceEngine;

extern const char REDUCE_AND[] = "ReduceAnd";
extern const char REDUCE_MIN[] = "ReduceMin";
extern const char REDUCE_MAX[] = "ReduceMax";
extern const char REDUCE_SUM[] = "ReduceSum";
extern const char REDUCE_MEAN[] = "ReduceMean";

template <typename DataType> class ConstantTraits;

template<> class ConstantTraits<ie_fp16>
{
public:
    static ie_fp16 one;
    static ie_fp16 zero;
};

template<> class ConstantTraits<int32_t>
{
public:
    static int32_t one;
    static int32_t zero;
};

template<> class ConstantTraits<float>
{
public:
    static float one;
    static float zero;
};

ie_fp16 ConstantTraits<ie_fp16>::one = PrecisionUtils::f32tof16(1.0f);
ie_fp16 ConstantTraits<ie_fp16>::zero = PrecisionUtils::f32tof16(0.0f);

int32_t ConstantTraits<int32_t>::one = 1;
int32_t ConstantTraits<int32_t>::zero = 0;

float ConstantTraits<float>::one = 1.0f;
float ConstantTraits<float>::zero = 0.0f;

template <typename Internal, typename External> Internal toInternal(External value) { return value; }
template <typename Internal, typename External> External toExternal(Internal value) { return value; }
template<> float toInternal<float, ie_fp16>(ie_fp16 val) { return PrecisionUtils::f16tof32(val); }
template<> ie_fp16 toExternal<float, ie_fp16>(float val) { return PrecisionUtils::f32tof16(val); }

template<typename DataType, typename DataTypeInternal=DataType>
class RefReduceAnd: public IReduceKernel<DataType>
{
    typedef ConstantTraits<DataType> Constants;
    typedef ConstantTraits<DataTypeInternal> InternalTypeConstants;
public:
    virtual void init() override
    { m_val = true; }
    virtual void accumulate(const DataType& val) override
    {
        auto actualVal = toInternal<DataTypeInternal, DataType>(val);
        m_val &= bool(actualVal != InternalTypeConstants::zero);
    }
    virtual DataType result() const override
    { return (m_val ? Constants::one : Constants::zero); }
    virtual DataType copy(const DataType& val) const override
    {
        auto actualVal = toInternal<DataTypeInternal, DataType>(val);
        return (actualVal != InternalTypeConstants::zero ? Constants::one : Constants::zero);
    }
    static void generateData(Blob::Ptr blob)
    {
        GenRandomData(blob);

        DataType* data = blob->buffer().as<DataType*>();
        const auto dims = blob->getTensorDesc().getDims();
        const int total = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int32_t>());

        // For consistent testing of ReduceAnd, we need to generate mostly non-zero input data,
        // with randomly placed 0s, but not in all reduced sub-tensors - not so often, not so seldom.

        for (int i = 0; i < total; ++i)
            data[i] = (data[i] == Constants::zero) ? Constants::one : data[i];
        for (int i = 0; i < total; i += 13)
            data[i] = Constants::zero;
    }
private:
    bool m_val;
};


template<typename DataType, typename DataTypeInternal=DataType>
class RefReduceMin: public IReduceKernel<DataType>
{
public:
    virtual void init() override
    { m_val = std::numeric_limits<DataTypeInternal>::max(); }
    virtual void accumulate(const DataType& val) override
    {
        DataTypeInternal fval = toInternal<DataTypeInternal, DataType>(val);
        m_val = m_val < fval ? m_val : fval;
    }
    virtual DataType result() const override
    { return toExternal<DataTypeInternal, DataType>(m_val); }
    virtual DataType copy(const DataType& val) const override
    { return val;}
    static void generateData(Blob::Ptr blob)
    {
        GenRandomData(blob);
    }
private:
    DataTypeInternal m_val;
};


template<typename DataType, typename DataTypeInternal=DataType>
class RefReduceMax: public IReduceKernel<DataType>
{
public:
    virtual void init() override
    {
        m_val = std::numeric_limits<DataTypeInternal>::lowest();
    }
    virtual void accumulate(const DataType& val) override
    {
        DataTypeInternal fval = toInternal<DataTypeInternal, DataType>(val);
        m_val = m_val > fval ? m_val : fval;
    }
    virtual DataType result() const override
    {
        return toExternal<DataTypeInternal, DataType>(m_val);
    }
    virtual DataType copy(const DataType& val) const override
    { return val;}
    static void generateData(Blob::Ptr blob) {
        GenRandomData(blob);
    }
private:
    DataTypeInternal m_val;
};

template<typename DataType, typename DataTypeInternal=DataType>
class RefReduceSum: public IReduceKernel<DataType>
{
public:
    virtual void init() override
    {
        m_val = 0;
    }
    virtual void accumulate(const DataType& val) override
    {
        m_val += toInternal<DataTypeInternal, DataType>(val);
    }
    virtual DataType result() const override
    {
        return toExternal<DataTypeInternal, DataType>(m_val);
    }
    virtual DataType copy(const DataType& val) const override
    { return val;}
    static void generateData(Blob::Ptr blob) {
        GenRandomData(blob);
    }
private:
    DataTypeInternal m_val;
};


template<typename DataType, typename DataTypeInternal=DataType>
class RefReduceMean: public IReduceKernel<DataType>
{
public:
    virtual void init() override
    {
        m_val = 0;
        m_count = 0;
    }
    virtual void accumulate(const DataType& val) override
    {
        m_val += toInternal<DataTypeInternal, DataType>(val);
        m_count++;
    }
    virtual DataType result() const override
    {
        if (m_count == 0) {
            return toExternal<DataTypeInternal, DataType>(m_val);
        } else {
            return toExternal<DataTypeInternal, DataType>(m_val / static_cast<DataTypeInternal>(m_count));
        }
    }
    virtual DataType copy(const DataType& val) const override
    { return val;}
    static void generateData(Blob::Ptr blob) {
        GenRandomData(blob);
    }
private:
    DataTypeInternal m_val;
    size_t m_count;
};

static RefReduceAnd<ie_fp16, float> refReduceAndFP16;
static RefReduceMin<ie_fp16, float> refReduceMinFP16;
static RefReduceMax<ie_fp16, float> refReduceMaxFP16;
static RefReduceSum<ie_fp16, float> refReduceSumFP16;
static RefReduceMean<ie_fp16, float> refReduceMeanFP16;

static RefReduceAnd<int32_t> refReduceAndI32;
static RefReduceMin<int32_t> refReduceMinI32;
static RefReduceMax<int32_t> refReduceMaxI32;
static RefReduceSum<int32_t> refReduceSumI32;
static RefReduceMean<int32_t> refReduceMeanI32;

typedef void GenData(Blob::Ptr blob);

template <typename DataType>
struct ReduceOpParams
{
    IReduceKernel<DataType>* op;
    float compare_threshold;
    GenData* generateData;
};

static const std::map<const char*, ReduceOpParams<ie_fp16>> refMapFP16 =
        {
                {REDUCE_AND, {&refReduceAndFP16, 0.0f, RefReduceAnd<ie_fp16>::generateData}},
                {REDUCE_MIN, {&refReduceMinFP16, 0.0f, RefReduceMin<ie_fp16>::generateData}},
                {REDUCE_MAX, {&refReduceMaxFP16, 0.0f, RefReduceMax<ie_fp16>::generateData}},
                {REDUCE_SUM, {&refReduceSumFP16, 0.01f, RefReduceSum<ie_fp16>::generateData}},
                {REDUCE_MEAN, {&refReduceMeanFP16, 0.01f, RefReduceMean<ie_fp16>::generateData}},
        };

static const std::map<const char*, ReduceOpParams<int32_t>> refMapI32 =
        {
                {REDUCE_AND, {&refReduceAndI32,  0.0f,  RefReduceAnd<int32_t>::generateData}},
                {REDUCE_MIN, {&refReduceMinI32,  0.0f,  RefReduceMin<int32_t>::generateData}},
                {REDUCE_MAX, {&refReduceMaxI32,  0.0f,  RefReduceMax<int32_t>::generateData}},
                {REDUCE_SUM, {&refReduceSumI32,  0.0f,  RefReduceSum<int32_t>::generateData}},
                {REDUCE_MEAN, {&refReduceMeanI32, 0.0f, RefReduceMean<int32_t>::generateData}},
        };

using ReduceTestParams = std::tuple<std::pair<SizeVector, vpu::LayoutPreference>, SizeVector, Precision, bool>;

static const Precision axesPrecision = Precision::I32;

class ReduceUtils
{
public:
    static std::string getModel(const SizeVector& inputDims, const SizeVector& axesList,
                                const SizeVector& outputDims, const std::string& reduceType,
                                const Precision dataPrecision, int keep_dims)
    {
        std::string model = R"V0G0N(
                <net name="testReduce" version="5">
                    <layers>
                        <layer id="0" name="reduce_input" precision="__DATA_PRECISION__" type="Input">
                            <output>
                                <port id="0">__INPUT_DIMS__</port>
                            </output>
                        </layer>
                        <layer id="1" name="reduce_axes" precision="__AXES_PRECISION__" type="Const">
                            <output>
                                <port id="1">__AXES_DIMS__</port>
                            </output>
                            <blobs>
                                <custom offset="0" size="__AXES_SIZE__"/>
                            </blobs>
                        </layer>
                        <layer id="2" name="reduce" precision="__DATA_PRECISION__" type="__REDUCE_TYPE__">
                            <data keep_dims="__KEEP_DIMS__"/>
                            <input>
                                <port id="0">__INPUT_DIMS__</port>
                                <port id="1">__AXES_DIMS__</port>
                            </input>
                            <output>
                                <port id="2">__OUTPUT_DIMS__</port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                    </edges>
                </net>
            )V0G0N";

        const SizeVector axesDims = { axesList.size() };

        std::string input_dims = dimsToString(inputDims);
        std::string axes_dims = dimsToString(axesDims);
        std::string output_dims = dimsToString(outputDims);
        size_t axes_size = axesList.size() * sizeof(int32_t);

        REPLACE_WITH_STR(model, "__REDUCE_TYPE__", reduceType);
        REPLACE_WITH_STR(model, "__DATA_PRECISION__", dataPrecision.name());
        REPLACE_WITH_STR(model, "__AXES_PRECISION__", axesPrecision.name());
        REPLACE_WITH_STR(model, "__INPUT_DIMS__", input_dims);
        REPLACE_WITH_STR(model, "__AXES_DIMS__", axes_dims);
        REPLACE_WITH_NUM(model, "__AXES_SIZE__", axes_size);
        REPLACE_WITH_STR(model, "__OUTPUT_DIMS__", output_dims);
        REPLACE_WITH_STR(model, "__KEEP_DIMS__", (keep_dims ? "True" : "False"));

        return model;
    }
    static std::string dimsToString(const SizeVector& dims)
    {
        std::string str;
        for (auto& d : dims)
            str += "<dim>" + std::to_string(d) + "</dim>";
        return str;
    }
    static SizeVector calcOutputDims(const SizeVector& inputDims, const SizeVector& axesList, int keep_dims)
    {
        auto mask = list2mask(inputDims.size(), axesList);
        if (keep_dims)
        {
            SizeVector outputDims(inputDims.size(), 0);
            for (int i = 0; i < (int)inputDims.size(); ++i)
            {
                if (mask & (1 << i))
                    outputDims[i] = 1;
                else
                    outputDims[i] = inputDims[i];
            }
            return outputDims;
        }
        else
        {
            SizeVector outputDims;
            for (int i = 0; i < (int)inputDims.size(); ++i)
            {
                if (!(mask & (1 << i)))
                    outputDims.push_back(inputDims[i]);
            }
            if (!(outputDims.size() > 0)) // 0D -> 1D
                outputDims.push_back(1);
            return outputDims;
        }
    }
    static unsigned list2mask(int ndims, const SizeVector& list)
    {
        unsigned mask = 0;
        for (int i : list)
        {
            if (i < 0) // handle negative indices
                i = ndims - std::abs(i);
            EXPECT_TRUE((i >= 0) && (i < ndims));
            mask |= (1 << i);
        }
        return mask;
    }

    static void getAxesBlob(const SizeVector& axesList, TBlob<uint8_t>::Ptr& weightsBlob, TBlob<int32_t>::Ptr& axesBlob)
    {
        size_t axes_size = axesList.size();
        size_t weights_size = axesList.size() * sizeof(int32_t);

        TBlob<uint8_t>* weights_raw = new TBlob<uint8_t>(TensorDesc(Precision::U8, {weights_size}, C));
        weights_raw->allocate();
        int32_t* weightsData = weights_raw->data().as<int32_t*>();

        TBlob<int32_t>* axes_raw = new TBlob<int32_t>(TensorDesc(Precision::I32, {axes_size}, C));
        axes_raw->allocate();
        int32_t* axesData = axes_raw->data().as<int32_t*>();

        for (size_t index = 0; index < axesList.size(); ++index) {
            weightsData[index] = axesList[index];
            axesData[index] = axesList[index];
        }

        weightsBlob = TBlob<uint8_t>::Ptr(weights_raw);
        axesBlob = TBlob<int32_t>::Ptr(axes_raw);
    }
};

template <const char* ReduceType>
class ReduceTest: public myriadLayerTestBaseWithParam<ReduceTestParams>
{
protected:
    void testReduce()
    {
        DISABLE_IF(!CheckMyriadX());
        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        const auto params = GetParam();
        const auto inputPair = std::get<0>(params);
        auto axesList = std::get<1>(params);
        const auto dataPrecision = std::get<2>(params);
        const int keepDims = std::get<3>(params) ? 1 : 0;

        const auto inputDims = inputPair.first;
        const auto layoutPreference = inputPair.second;

        const auto outputDims = ReduceUtils::calcOutputDims(inputDims, axesList, keepDims);
        const auto model = ReduceUtils::getModel(inputDims, axesList, outputDims, ReduceType, dataPrecision, keepDims);

        TBlob<uint8_t>::Ptr weightsBlob;
        TBlob<int32_t>::Ptr axesBlob;
        ReduceUtils::getAxesBlob(axesList, weightsBlob, axesBlob);
        ASSERT_NE(weightsBlob, nullptr);
 
        ASSERT_NO_THROW(readNetwork(model, weightsBlob));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["reduce_input"]->setPrecision(dataPrecision);
        _inputsInfo["reduce_input"]->setLayout(vpu::deviceLayout(TensorDesc::getLayoutByDims(inputDims), layoutPreference));

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["reduce"]->setPrecision(dataPrecision);
        _outputsInfo["reduce"]->setLayout(vpu::deviceLayout(TensorDesc::getLayoutByDims(outputDims), layoutPreference));

        ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, _config));
        ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
        
        Blob::Ptr inputBlob;
        ASSERT_NO_THROW(inputBlob = _inferRequest.GetBlob("reduce_input"));
        
        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = _inferRequest.GetBlob("reduce"));
        
        Blob::Ptr refBlob = nullptr;
        float compareThreshold = 0.0f;
        if (dataPrecision == Precision::FP16) {
            auto opIt = refMapFP16.find(ReduceType);
            ASSERT_TRUE(opIt != refMapFP16.end());
            compareThreshold = opIt->second.compare_threshold;
            auto reduceOp = opIt->second.op;
            auto generateData = opIt->second.generateData;
            generateData(inputBlob);

            ASSERT_NO_THROW(_inferRequest.Infer());
            
            refBlob = make_shared_blob<ie_fp16>(outputBlob->getTensorDesc());
            refBlob->allocate();
            ref_reduce(inputBlob, axesBlob, refBlob, keepDims, layoutPreference, reduceOp);
            CompareCommonAbsolute(outputBlob, refBlob, compareThreshold);
       } else if (dataPrecision == Precision::I32) {
           auto opIt = refMapI32.find(ReduceType);
           ASSERT_TRUE(opIt != refMapI32.end());
           auto reduceOp = opIt->second.op;
           auto generateData = opIt->second.generateData;
           generateData(inputBlob);

           ASSERT_NO_THROW(_inferRequest.Infer());
           
           refBlob = make_shared_blob<int32_t>(outputBlob->getTensorDesc());
           refBlob->allocate();
           ref_reduce(inputBlob, axesBlob, refBlob, keepDims, layoutPreference, reduceOp);
           CompareCommonExact(outputBlob, refBlob);
       }
    }
};

using myriadTestsReduceAnd_smoke = ReduceTest<REDUCE_AND>;
using myriadTestsReduceMin_smoke = ReduceTest<REDUCE_MIN>;
using myriadTestsReduceMax_smoke = ReduceTest<REDUCE_MAX>;
using myriadTestsReduceSum_smoke = ReduceTest<REDUCE_SUM>;
using myriadTestsReduceMean_smoke = ReduceTest<REDUCE_MEAN>;

// Tests are disabled due to hang: #-28315

TEST_P(myriadTestsReduceAnd_smoke, And)
{
    testReduce();
}
TEST_P(myriadTestsReduceMin_smoke, Min)
{
    testReduce();
}
TEST_P(myriadTestsReduceMax_smoke, Max)
{
    testReduce();
}
TEST_P(myriadTestsReduceSum_smoke, Sum)
{
    testReduce();
}
TEST_P(myriadTestsReduceMean_smoke, Mean)
{
    testReduce();
}

static const std::vector<std::pair<SizeVector, vpu::LayoutPreference>> s_input_pair =
        {
                {{1, 3, 2, 14, 32}, vpu::LayoutPreference::ChannelMinor},
                {{1, 3, 2, 14, 32}, vpu::LayoutPreference::ChannelMajor},
                {{2, 2, 2, 14, 32}, vpu::LayoutPreference::ChannelMinor},
                {{2, 2, 2, 14, 32}, vpu::LayoutPreference::ChannelMajor},
                {{3, 5, 4, 8, 16}, vpu::LayoutPreference::ChannelMinor},
                {{3, 5, 4, 8, 16}, vpu::LayoutPreference::ChannelMajor},
                {{4, 2, 16, 16, 8}, vpu::LayoutPreference::ChannelMinor},
                {{4, 2, 16, 16, 8}, vpu::LayoutPreference::ChannelMajor},

                {{3, 2, 14, 32}, vpu::LayoutPreference::ChannelMinor},
                {{3, 2, 14, 32}, vpu::LayoutPreference::ChannelMajor},
                {{2, 2, 14, 32}, vpu::LayoutPreference::ChannelMinor},
                {{2, 2, 14, 32}, vpu::LayoutPreference::ChannelMajor},
                {{5, 4, 8, 16}, vpu::LayoutPreference::ChannelMinor},
                {{5, 4, 8, 16}, vpu::LayoutPreference::ChannelMajor},
                {{2, 16, 16, 8}, vpu::LayoutPreference::ChannelMinor},
                {{2, 16, 16, 8}, vpu::LayoutPreference::ChannelMajor},

                {{3, 2, 14}, vpu::LayoutPreference::ChannelMajor},
                {{2, 2, 14}, vpu::LayoutPreference::ChannelMajor},
                {{5, 4, 8}, vpu::LayoutPreference::ChannelMajor},
                {{2, 16, 16}, vpu::LayoutPreference::ChannelMajor},

                {{7, 3, 5, 1, 7, 11, 12}, vpu::LayoutPreference::ChannelMajor},
        };

static const std::vector<SizeVector> s_axes_list =
        {
                {1},
                {0, 2},
                {0, 1, 2},
        };

static const std::vector<Precision> s_data_precision =
        {
                Precision::FP16,
                Precision::I32
        };

static const std::vector<bool> s_keep_dims =
        {
                false,
                true,
        };
