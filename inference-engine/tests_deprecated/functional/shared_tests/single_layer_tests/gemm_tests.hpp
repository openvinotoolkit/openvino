// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include <single_layer_common.hpp>
#include <string>
#include <tuple>

using namespace InferenceEngine;

struct gemm_base_params {
    float alpha;
    float beta;
    bool transpose_A;
    bool transpose_B;
    SizeVector dims_A;
    SizeVector dims_B;
    SizeVector dims_C;

    gemm_base_params() = default;
    gemm_base_params(float _alpha,
                     float _beta,
                     bool _transpose_A,
                     bool _transpose_B,
                     SizeVector _dims_A,
                     SizeVector _dims_B,
                     SizeVector _dims_C = {})
        : alpha(_alpha)
        , beta(_beta)
        , transpose_A(_transpose_A)
        , transpose_B(_transpose_B)
        , dims_A(_dims_A)
        , dims_B(_dims_B)
        , dims_C(_dims_C)
    {}

    virtual void print(std::ostream& os) const {
        os << "alpha: " << alpha << ", beta: " << beta
            << ", trans A: " << transpose_A << ", trans B: " << transpose_B
            << std::endl;

        auto print_dims = [&](std::string name, const SizeVector& dims) {
            os << name << ": {";
            if (!dims.empty())
                os << dims[0];
            for (size_t i = 1; i < dims.size(); ++i)
                os << ", " << dims[i];
            os << "}" << std::endl;
        };

        print_dims("A", dims_A);
        print_dims("B", dims_B);
        print_dims("C", dims_C);
    }

    virtual SizeVector outDims() const {
        size_t max_dims_num = std::max(dims_A.size(), dims_B.size());
        max_dims_num = std::max(dims_C.size(), max_dims_num);

        SizeVector dims_out(max_dims_num);
        // Process batch dims in reverse for required alignment
        for (size_t rbi = 0; rbi < max_dims_num - 2; ++rbi) {
            size_t max_val = 1;

            if (rbi + 2 < dims_A.size()) {
                auto bi_A = dims_A.size() - rbi - 3;
                max_val = std::max(max_val, dims_A[bi_A]);
            }
            if (rbi + 2 < dims_B.size()) {
                auto bi_B = dims_B.size() - rbi - 3;
                max_val = std::max(max_val, dims_B[bi_B]);
            }
            if (rbi + 2 < dims_C.size()) {
                auto bi_C = dims_C.size() - rbi - 3;
                max_val = std::max(max_val, dims_C[bi_C]);
            }

            auto bi_out = max_dims_num - rbi - 3;
            dims_out[bi_out] = max_val;
        }

        auto y_dim_A = transpose_A ? dims_A.size() - 1 : dims_A.size() - 2;
        auto x_dim_B = transpose_B ? dims_B.size() - 2 : dims_B.size() - 1;
        dims_out[dims_out.size() - 1] = dims_B[x_dim_B];
        dims_out[dims_out.size() - 2] = dims_A[y_dim_A];

        return dims_out;
    }
};


std::vector<float> ref_gemm(const gemm_base_params& params,
                            const std::vector<float>& data_A,
                            const std::vector<float>& data_B,
                            const std::vector<float>& data_C) {
    const auto& dims_A = params.dims_A;
    const auto& dims_B = params.dims_B;
    const auto& dims_C = params.dims_C;

    bool use_C = !dims_C.empty();

    auto x_A = dims_A[dims_A.size() - 1];
    auto y_A = dims_A[dims_A.size() - 2];
    auto x_pitch_A = size_t(1);
    auto y_pitch_A = x_A;

    auto x_B = dims_B[dims_B.size() - 1];
    auto y_B = dims_B[dims_B.size() - 2];
    auto x_pitch_B = size_t(1);
    auto y_pitch_B = x_B;

    if (params.transpose_A) {
        std::swap(x_A, y_A);
        std::swap(x_pitch_A, y_pitch_A);
    }

    if (params.transpose_B) {
        std::swap(x_B, y_B);
        std::swap(x_pitch_B, y_pitch_B);
    }

    auto dims_out = params.outDims();

    auto x_out = dims_out[dims_out.size() - 1];
    auto y_out = dims_out[dims_out.size() - 2];
    auto x_pitch_out = size_t(1);
    auto y_pitch_out = x_out;

    auto out_batch_num = dims_out.size() - 2;

    // Calculates batch pitches in reverse order
    auto calculate_batch_pitches = [out_batch_num](const SizeVector& dims) {
        std::vector<size_t> batch_pitches = { };
        batch_pitches.reserve(out_batch_num);
        size_t real_pitch = dims[dims.size() - 2] * dims[dims.size() - 1];

        for (size_t rbi = 0; rbi < out_batch_num; ++rbi) {
            if (rbi + 2 < dims.size() && dims[dims.size() - rbi - 3] != 1) {
                batch_pitches.push_back(real_pitch);
                real_pitch *= dims[dims.size() - rbi - 3];
            } else {
                // Set to zero for broadcasting
                batch_pitches.push_back(0ul);
            }
        }

        return batch_pitches;
    };

    auto batch_pitches_A = calculate_batch_pitches(dims_A);
    auto batch_pitches_B = calculate_batch_pitches(dims_B);
    auto batch_pitches_C = use_C ? calculate_batch_pitches(dims_C) : std::vector<size_t>();
    auto batch_pitches_out = calculate_batch_pitches(dims_out);

    auto k = x_A;

    auto total_out_size = std::accumulate(dims_out.begin(), dims_out.end(), 1ul, std::multiplies<size_t>());
    std::vector<float> data_out(total_out_size, 0.f);

    // Currently processed batch indices in reverse order
    std::vector<size_t> current_batch_indices(out_batch_num, 0ul);
    auto get_current_batch_offset = [&](const std::vector<size_t>& pitches) {
        return std::inner_product(pitches.begin(), pitches.end(), current_batch_indices.begin(), 0ul);
    };

    do {
        auto batch_offset_A = get_current_batch_offset(batch_pitches_A);
        auto batch_offset_B = get_current_batch_offset(batch_pitches_B);
        auto batch_offset_C = use_C ? get_current_batch_offset(batch_pitches_C) : 0ul;
        auto batch_offset_out = get_current_batch_offset(batch_pitches_out);

        for (size_t yi = 0; yi < y_out; ++yi) {
            for (size_t xi = 0; xi < x_out; ++xi) {

                float acc = 0.f;
                if (params.alpha != 0.f) {
                    for (size_t ki = 0; ki < k; ++ki) {
                        auto idx_A = batch_offset_A + yi * y_pitch_A + ki * x_pitch_A;
                        auto idx_B = batch_offset_B + ki * y_pitch_B + xi * x_pitch_B;

                        acc += data_A[idx_A] * data_B[idx_B];
                    }

                    acc *= params.alpha;
                }

                if (use_C && params.beta != 0.f) {
                    auto idx_C = batch_offset_C + yi * y_pitch_out + xi * x_pitch_out;
                    acc += params.beta * data_C[idx_C];
                }

                auto idx_out = batch_offset_out + yi * y_pitch_out + xi * x_pitch_out;
                data_out[idx_out] = acc;
            }
        }

        for (size_t i = 0; i < out_batch_num; ++i) {
            current_batch_indices[i] += 1;
            if (current_batch_indices[i] == dims_out[dims_out.size() - 3 - i] &&
                i != out_batch_num - 1) {  // Don't reset last index as it signals end of calculations
                current_batch_indices[i] = 0;
            } else {
                break;
            }
        }
    } while (current_batch_indices.size() > 0 &&
             current_batch_indices[current_batch_indices.size() - 1] != dims_out[0]);

    return data_out;
}

struct gemm_test_params : gemm_base_params {
    std::string device_name;
    std::string precision;

    gemm_test_params(std::string name, std::string _precision, gemm_base_params base)
        : gemm_base_params(base)
        , device_name(name)
        , precision(_precision)
    {}

    gemm_test_params(std::tuple<std::string, std::string, gemm_base_params> wrapper)
        : gemm_test_params(std::get<0>(wrapper), std::get<1>(wrapper), std::get<2>(wrapper))
    {}

    void print(std::ostream& os) const override {
        os << "Device: " << device_name << ", precision: " << precision << std::endl;
        gemm_base_params::print(os);
    }
};

class GemmTestBase : public TestsCommon {
    std::string model_t = R"V0G0N(
<net name="GemmSingleLayerTest" version="5" precision="_PRECISION_" batch="1">
    <layers>
        <layer name="input_A" type="Input" id="1" precision="_PRECISION_">
            <output>
                <port id="0">
                    _IN_A_DIMS_
                </port>
            </output>
        </layer>
        <layer name="input_B" type="Input" id="2" precision="_PRECISION_">
            <output>
                <port id="0">
                    _IN_B_DIMS_
                </port>
            </output>
        </layer>
        _IN_C_LAYER_
        <layer name="gemm" type="GEMM" id="4" precision="_PRECISION_">
            <data alpha="_ALPHA_" beta="_BETA_" transpose_a="_TRANS_A_" transpose_b="_TRANS_B_" />
            <input>
                <port id="0">
                    _IN_A_DIMS_
                </port>
                <port id="1">
                    _IN_B_DIMS_
                </port>
                _IN_C_GEMM_PORT_
            </input>
            <output>
                <port id="0">
                    _OUT_DIMS_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="1"/>
        _IN_C_EDGE_
    </edges>
</net>
)V0G0N";

std::string in_C_layer = R"V0G0N(
        <layer name="input_C" type="Input" id="3" precision="_PRECISION_">
            <output>
                <port id="0">
                    _IN_C_DIMS_
                </port>
            </output>
        </layer>
)V0G0N";

std::string in_C_port = R"V0G0N(
                <port id="2">
                    _IN_C_DIMS_
                </port>
)V0G0N";

std::string in_C_edge = R"V0G0N(
        <edge from-layer="3" from-port="0" to-layer="4" to-port="2"/>
)V0G0N";

protected:
    virtual float getThreshold(const gemm_test_params& params) {
        if (params.precision == "FP16")
            return 0.02f;
        else
            return 0.01f;
    }

    std::string getModel(const gemm_test_params& params) {
        auto model = model_t;

        if (!params.dims_C.empty()) {
            REPLACE_WITH_STR(model, "_IN_C_LAYER_", in_C_layer);
            REPLACE_WITH_STR(model, "_IN_C_GEMM_PORT_", in_C_port);
            REPLACE_WITH_STR(model, "_IN_C_EDGE_", in_C_edge);
        } else {
            REPLACE_WITH_STR(model, "_IN_C_LAYER_", "");
            REPLACE_WITH_STR(model, "_IN_C_GEMM_PORT_", "");
            REPLACE_WITH_STR(model, "_IN_C_EDGE_", "");
        }

        REPLACE_WITH_STR(model, "_PRECISION_", params.precision);

        REPLACE_WITH_NUM(model, "_ALPHA_", params.alpha);
        REPLACE_WITH_NUM(model, "_BETA_", params.beta);
        REPLACE_WITH_NUM(model, "_TRANS_A_", params.transpose_A);
        REPLACE_WITH_NUM(model, "_TRANS_B_", params.transpose_B);

        auto get_dims_str = [](const SizeVector& dims) {
            std::string result;
            for (const auto& d : dims) {
                result += "<dim>" + std::to_string(d) + "</dim>\n";
            }
            return result;
        };

        std::string in_A_dims = get_dims_str(params.dims_A);
        std::string in_B_dims = get_dims_str(params.dims_B);
        std::string in_C_dims = get_dims_str(params.dims_C);
        std::string out_dims = get_dims_str(params.outDims());

        REPLACE_WITH_STR(model, "_IN_A_DIMS_", in_A_dims);
        REPLACE_WITH_STR(model, "_IN_B_DIMS_", in_B_dims);
        REPLACE_WITH_STR(model, "_IN_C_DIMS_", in_C_dims);
        REPLACE_WITH_STR(model, "_OUT_DIMS_", out_dims);

        return model;
    }

    CNNNetwork getNetwork(Core & ie, const gemm_test_params& params) {
        std::string model = getModel(params);

        CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());

        network.getInputsInfo().at("input_A")->setPrecision(Precision::FP32);
        network.getInputsInfo().at("input_B")->setPrecision(Precision::FP32);
        if (!params.dims_C.empty())
            network.getInputsInfo().at("input_C")->setPrecision(Precision::FP32);

        network.getOutputsInfo().at("gemm")->setPrecision(Precision::FP32);

        return network;
    }

    void runTest(const gemm_test_params& test_params,
                 const std::vector<float>& data_A,
                 const std::vector<float>& data_B,
                 const std::vector<float>& data_C,
                 const std::vector<float>& ref_output) {
        test_params.print(std::cout);

        Core ie;
        auto network = getNetwork(ie, test_params);
        auto exec = ie.LoadNetwork(network, test_params.device_name);
        auto request = exec.CreateInferRequest();

        auto fill_blob = [&](const char* name, const std::vector<float>& data) {
            Blob::Ptr blob = request.GetBlob(name);

            auto fill_size = std::min(blob->size(), data.size());
            auto buffer = blob->buffer().as<float*>();

            for (size_t i = 0; i < fill_size; ++i) {
                buffer[i] = data[i];
            }
        };

        fill_blob("input_A", data_A);
        fill_blob("input_B", data_B);
        if (!test_params.dims_C.empty()) {
            fill_blob("input_C", data_C);
        }

        request.Infer();

        if (!ref_output.empty()) {
            Blob::Ptr blob_out = request.GetBlob("gemm");
            ASSERT_EQ(blob_out->size(), ref_output.size());

            auto buf_out = blob_out->buffer().as<float*>();
            compare(buf_out, ref_output.data(), blob_out->size(), getThreshold(test_params));
        }
    }
};

using GemmRandomTestParam = std::tuple<
    std::string,        // plugin
    std::string,        // precision
    gemm_base_params>;  // gemm params

class GemmRandomTest : public GemmTestBase, public testing::WithParamInterface<GemmRandomTestParam> {};

// Basic cases: all transposition combinations, 2D-5D
#define case1  gemm_base_params(1.2f, 3.f,   false, false, {9ul, 11ul}, {11ul, 13ul} )
#define case2  gemm_base_params(1.2f, 3.f,   false, false, {9ul, 11ul}, {11ul, 13ul}, {9ul, 13ul} )
#define case3  gemm_base_params(2.5f, 1.2f,  false, false, {7ul, 9ul, 11ul}, {7ul, 11ul, 13ul} )
#define case4  gemm_base_params(2.5f, 1.2f,  false, false, {7ul, 9ul, 11ul}, {7ul, 11ul, 13ul}, {7ul, 9ul, 13ul} )
#define case5  gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul})
#define case6  gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul}, {3ul, 7ul, 9ul, 13ul} )
#define case7  gemm_base_params(1.2f, -1.5f, false, false, {2ul, 3ul, 7ul, 9ul, 11ul}, {2ul, 3ul, 7ul, 11ul, 13ul})
#define case8  gemm_base_params(1.2f, -1.5f, false, false, {2ul, 3ul, 7ul, 9ul, 11ul}, {2ul, 3ul, 7ul, 11ul, 13ul}, {2ul, 3ul, 7ul, 9ul, 13ul})
#define case9  gemm_base_params(1.2f, 3.f,   true,  false, {11ul, 9ul}, {11ul, 13ul} )
#define case10 gemm_base_params(1.2f, 3.f,   true,  false, {11ul, 9ul}, {11ul, 13ul}, {9ul, 13ul} )
#define case11 gemm_base_params(2.5f, 1.2f,  true,  false, {7ul, 11ul, 9ul}, {7ul, 11ul, 13ul} )
#define case12 gemm_base_params(2.5f, 1.2f,  true,  false, {7ul, 11ul, 9ul}, {7ul, 11ul, 13ul}, {7ul, 9ul, 13ul} )
#define case13 gemm_base_params(1.2f, -1.5f, true,  false, {3ul, 7ul, 11ul, 9ul}, {3ul, 7ul, 11ul, 13ul})
#define case14 gemm_base_params(1.2f, -1.5f, true,  false, {3ul, 7ul, 11ul, 9ul}, {3ul, 7ul, 11ul, 13ul}, {3ul, 7ul, 9ul, 13ul} )
#define case15 gemm_base_params(1.2f, -1.5f, true,  false, {2ul, 3ul, 7ul, 11ul, 9ul}, {2ul, 3ul, 7ul, 11ul, 13ul})
#define case16 gemm_base_params(1.2f, -1.5f, true,  false, {2ul, 3ul, 7ul, 11ul, 9ul}, {2ul, 3ul, 7ul, 11ul, 13ul}, {2ul, 3ul, 7ul, 9ul, 13ul})
#define case17 gemm_base_params(1.2f, 3.f,   false, true,  {9ul, 11ul}, {13ul, 11ul} )
#define case18 gemm_base_params(1.2f, 3.f,   false, true,  {9ul, 11ul}, {13ul, 11ul}, {9ul, 13ul} )
#define case19 gemm_base_params(2.5f, 1.2f,  false, true,  {7ul, 9ul, 11ul}, {7ul, 13ul, 11ul} )
#define case20 gemm_base_params(2.5f, 1.2f,  false, true,  {7ul, 9ul, 11ul}, {7ul, 13ul, 11ul}, {7ul, 9ul, 13ul} )
#define case21 gemm_base_params(1.2f, -1.5f, false, true,  {3ul, 7ul, 9ul, 11ul}, {3ul, 7ul, 13ul, 11ul})
#define case22 gemm_base_params(1.2f, -1.5f, false, true,  {3ul, 7ul, 9ul, 11ul}, {3ul, 7ul, 13ul, 11ul}, {3ul, 7ul, 9ul, 13ul} )
#define case23 gemm_base_params(1.2f, -1.5f, false, true,  {2ul, 3ul, 7ul, 9ul, 11ul}, {2ul, 3ul, 7ul, 13ul, 11ul})
#define case24 gemm_base_params(1.2f, -1.5f, false, true,  {2ul, 3ul, 7ul, 9ul, 11ul}, {2ul, 3ul, 7ul, 13ul, 11ul}, {2ul, 3ul, 7ul, 9ul, 13ul})
#define case25 gemm_base_params(1.2f, 3.f,   true,  true,  {11ul, 9ul}, {13ul, 11ul} )
#define case26 gemm_base_params(1.2f, 3.f,   true,  true,  {11ul, 9ul}, {13ul, 11ul}, {9ul, 13ul} )
#define case27 gemm_base_params(2.5f, 1.2f,  true,  true,  {7ul, 11ul, 9ul}, {7ul, 13ul, 11ul} )
#define case28 gemm_base_params(2.5f, 1.2f,  true,  true,  {7ul, 11ul, 9ul}, {7ul, 13ul, 11ul}, {7ul, 9ul, 13ul} )
#define case29 gemm_base_params(1.2f, -1.5f, true,  true,  {3ul, 7ul, 11ul, 9ul}, {3ul, 7ul, 13ul, 11ul})
#define case30 gemm_base_params(1.2f, -1.5f, true,  true,  {3ul, 7ul, 11ul, 9ul}, {3ul, 7ul, 13ul, 11ul}, {3ul, 7ul, 9ul, 13ul} )
#define case31 gemm_base_params(1.2f, -1.5f, true,  true,  {2ul, 3ul, 7ul, 11ul, 9ul}, {2ul, 3ul, 7ul, 13ul, 11ul})
#define case32 gemm_base_params(1.2f, -1.5f, true,  true,  {2ul, 3ul, 7ul, 11ul, 9ul}, {2ul, 3ul, 7ul, 13ul, 11ul}, {2ul, 3ul, 7ul, 9ul, 13ul})

// Broadcasting/dimension inference cases
#define case33 gemm_base_params(1.2f, -1.5f, false, false, {1ul, 1ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul})
#define case34 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {1ul, 1ul, 11ul, 13ul})
#define case35 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul}, {1ul, 1ul, 9ul, 13ul})
#define case36 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 1ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul})
#define case37 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {3ul, 1ul, 11ul, 13ul})
#define case38 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul}, {3ul, 1ul, 9ul, 13ul})
#define case39 gemm_base_params(1.2f, -1.5f, false, false, {9ul, 11ul}, {3ul, 7ul, 11ul, 13ul})
#define case40 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {11ul, 13ul})
#define case41 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul}, {9ul, 13ul})
#define case42 gemm_base_params(1.2f, -1.5f, false, false, {7ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul})
#define case43 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {7ul, 11ul, 13ul})
#define case44 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 7ul, 9ul, 11ul}, {3ul, 7ul, 11ul, 13ul}, {7ul, 9ul, 13ul})
#define case45 gemm_base_params(1.2f, -1.5f, false, false, {7ul, 9ul, 11ul}, {3ul, 1ul, 11ul, 13ul})
#define case46 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 1ul, 9ul, 11ul}, {7ul, 11ul, 13ul})
#define case47 gemm_base_params(1.2f, -1.5f, false, false, {3ul, 1ul, 9ul, 11ul}, {3ul, 1ul, 11ul, 13ul}, {7ul, 9ul, 13ul})

#define all_cases                                                       \
    case1,  case2,  case3,  case4,  case5,  case6,  case7,  case8,      \
    case9,  case10, case11, case12, case13, case14, case15, case16,     \
    case17, case18, case19, case20, case21, case22, case23, case24,     \
    case25, case26, case27, case28, case29, case30, case31, case32,     \
    case33, case34, case35, case36, case37, case38,                     \
    case39, case40, case41, case42, case43, case44,                     \
    case45, case46, case47

TEST_P(GemmRandomTest, smoke_randomInput) {
    gemm_test_params params = GetParam();

    auto size_A = std::accumulate(params.dims_A.begin(), params.dims_A.end(), size_t(1), std::multiplies<size_t>());
    auto size_B = std::accumulate(params.dims_B.begin(), params.dims_B.end(), size_t(1), std::multiplies<size_t>());
    auto size_C = std::accumulate(params.dims_C.begin(), params.dims_C.end(), size_t(1), std::multiplies<size_t>());

    std::vector<float> data_A(size_A);
    std::vector<float> data_B(size_B);
    std::vector<float> data_C(size_C);

    fill_data(data_A.data(), size_A);
    fill_data(data_B.data(), size_B);
    fill_data(data_C.data(), size_C);

    auto ref_output = ref_gemm(params, data_A, data_B, data_C);

    runTest(params, data_A, data_B, data_C, ref_output);
};
