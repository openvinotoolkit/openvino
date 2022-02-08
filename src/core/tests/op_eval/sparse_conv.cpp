// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/sparse_conv.hpp"

#include <string>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/openvino.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

std::shared_ptr<Function> create_sparse_conv(size_t num_inp_pos,
                                             size_t num_out_pos,
                                             size_t inp_channels,
                                             size_t out_channels) {
    auto feat = make_shared<op::Parameter>(element::f32, Shape{num_inp_pos, inp_channels});
    auto inp_pos = make_shared<op::Parameter>(element::f32, Shape{num_inp_pos, 3});
    auto out_pos = make_shared<op::Parameter>(element::f32, Shape{num_out_pos, 3});
    auto kernel = make_shared<op::Parameter>(element::f32, Shape{out_channels, inp_channels, 3, 3, 3});
    auto offset = make_shared<op::Parameter>(element::f32, Shape{3});
    auto voxel_size = make_shared<op::Parameter>(element::f32, Shape{1});
    auto conv = make_shared<op::v1::SparseConv>(feat, inp_pos, out_pos, kernel, offset, voxel_size);
    return make_shared<Function>(OutputVector{conv},
                                 ParameterVector{feat, inp_pos, out_pos, kernel, offset, voxel_size});
}

TEST(op_eval, sparse_conv_single_channel) {
    auto fun = create_sparse_conv(2, 2, 1, 1);

    std::vector<float> features{1.0f, 1.0f};
    std::vector<float> inpPos{1.46057f, 3.3381f, 0.504631f, 1.00087f, 2.48036f, 1.01154f};
    std::vector<float> kernel(3 * 3 * 3);
    std::iota(kernel.begin(), kernel.end(), 1);

    std::vector<float> offset(3, 0.0f);
    float voxelSize = 1.0f;

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{2, 1}, features),
                               make_host_tensor<element::Type_t::f32>(Shape{2, 3}, inpPos),
                               make_host_tensor<element::Type_t::f32>(Shape{2, 3}, inpPos),  // out_pos
                               make_host_tensor<element::Type_t::f32>(Shape{1, 1, 3, 3, 3}, kernel),
                               make_host_tensor<element::Type_t::f32>(Shape{3}, offset),
                               make_host_tensor<element::Type_t::f32>(Shape{1}, {voxelSize})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_data = read_vector<float>(result);
    EXPECT_NEAR(result_data[0], 34.f, 0.000001);
    EXPECT_NEAR(result_data[1], 22.f, 0.000001);
}

static size_t generatePoses(size_t numPos, size_t maxGridExtent, std::vector<float>& data) {
    std::set<tuple<int, int, int>> uniquePos;
    for (int i = 0; i < numPos; ++i) {
        int x = rand() % maxGridExtent;
        int y = rand() % maxGridExtent;
        int z = rand() % maxGridExtent;
        uniquePos.insert(std::make_tuple(x, y, z));
    }
    numPos = uniquePos.size();

    data.resize(numPos * 3);
    size_t i = 0;
    for (const auto& it : uniquePos) {
        data[i * 3] = std::get<0>(it);
        data[i * 3 + 1] = std::get<1>(it);
        data[i * 3 + 2] = std::get<2>(it);
        i += 1;
    }
    return numPos;
}

// Compare SparseConv with Conv3D
struct SparseConvTest : ::testing::TestWithParam<tuple<size_t, size_t, size_t>> {};
TEST_P(SparseConvTest, sparse_conv_like_conv3d) {
    const size_t ic = get<0>(GetParam());
    const size_t oc = get<1>(GetParam());
    const size_t kernelSz = get<2>(GetParam());
    static const size_t grid = 4;
    size_t numInpPos = 100;
    size_t numOutPos = 50;

    // Generate random unique positions.
    std::vector<float> inpPos, outPos;
    numInpPos = generatePoses(numInpPos, grid, inpPos);
    numOutPos = generatePoses(std::min(numOutPos, numInpPos), grid, outPos);

    // Generate input features and kernel
    std::vector<float> features(numInpPos * ic);
    std::vector<float> denseFeatures(pow(grid, 3) * ic, 0.0f);
    std::vector<float> kernel(oc * ic * pow(kernelSz, 3));
    std::vector<float> offset(3, 0.0f);
    float voxelSize = 1.0f;
    for (size_t n = 0; n < numInpPos; ++n) {
        int x = static_cast<int>(inpPos[n * 3]);
        int y = static_cast<int>(inpPos[n * 3 + 1]);
        int z = static_cast<int>(inpPos[n * 3 + 2]);
        for (size_t i = 0; i < ic; ++i) {
            float value = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            features[n * ic + i] = value;
            denseFeatures[((i * grid + z) * grid + y) * grid + x] = value;
        }
    }

    for (size_t i = 0; i < kernel.size(); ++i)
        kernel[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    // Define sparse conv
    auto sparseConv = create_sparse_conv(numInpPos, numOutPos, ic, oc);

    // Define dense Conv3D
    auto conv3d_inp = make_shared<op::Parameter>(element::f32, Shape{1, ic, grid, grid, grid});
    auto conv3d_kernel = make_shared<op::Parameter>(element::f32, Shape{oc, ic, kernelSz, kernelSz, kernelSz});
    auto conv3d = std::make_shared<op::v1::Convolution>(conv3d_inp,
                                                        conv3d_kernel,
                                                        ngraph::Strides({1, 1, 1}),  // strides
                                                        ngraph::CoordinateDiff({}),  // pad_begin
                                                        ngraph::CoordinateDiff({}),  // pad_end
                                                        ngraph::Strides({1, 1, 1}),  // dilations
                                                        ngraph::op::PadType::SAME_LOWER);

    // Evaluate SparseConv
    auto out = make_shared<HostTensor>();
    ASSERT_TRUE(sparseConv->evaluate(
        {out},
        {make_host_tensor<element::Type_t::f32>(Shape{numInpPos, ic}, features),
         make_host_tensor<element::Type_t::f32>(Shape{numInpPos, 3}, inpPos),
         make_host_tensor<element::Type_t::f32>(Shape{numOutPos, 3}, outPos),
         make_host_tensor<element::Type_t::f32>(Shape{oc, ic, kernelSz, kernelSz, kernelSz}, kernel),
         make_host_tensor<element::Type_t::f32>(Shape{3}, offset),
         make_host_tensor<element::Type_t::f32>(Shape{1}, {voxelSize})}));
    EXPECT_EQ(out->get_element_type(), element::f32);
    auto outData = read_vector<float>(out);

    // Run Conv3D
    ov::Core core;
    std::shared_ptr<ov::Model> model(new ov::Model({conv3d}));
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    infer_request.set_tensor(conv3d_inp,
                             ov::Tensor(element::f32, Shape{1, ic, grid, grid, grid}, denseFeatures.data()));
    infer_request.set_tensor(conv3d_kernel,
                             ov::Tensor(element::f32, Shape{oc, ic, kernelSz, kernelSz, kernelSz}, kernel.data()));
    infer_request.infer();

    auto ref = infer_request.get_tensor(conv3d);

    float* refData = ref.data<float>();
    for (int n = 0; n < numOutPos; ++n) {
        int x = static_cast<int>(outPos[n * 3]);
        int y = static_cast<int>(outPos[n * 3 + 1]);
        int z = static_cast<int>(outPos[n * 3 + 2]);
        for (int ch = 0; ch < oc; ++ch) {
            float refVal = refData[((ch * grid + z) * grid + y) * grid + x];
            float outVal = outData[n * oc + ch];
            EXPECT_NEAR(outVal, refVal, 0.000001);
        }
    }
}
INSTANTIATE_TEST_CASE_P(/**/,
                        SparseConvTest,
                        testing::Combine(
                            /*inp_channels*/ testing::Values(1, 3),
                            /*out_channels*/ testing::Values(1, 4),
                            /*kernel_size*/ testing::Values(3)));
