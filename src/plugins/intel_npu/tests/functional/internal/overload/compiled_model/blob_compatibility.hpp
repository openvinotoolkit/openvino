// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <base/ov_behavior_test_utils.hpp>

#include "common/npu_test_env_cfg.hpp"
#include "openvino/core/version.hpp"

// models generation
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"

namespace {

// intentinally recycled code for models generation if we want to compare with refs
std::shared_ptr<ov::Node> make_convolution(const ov::Output<ov::Node>& in,
                                           const ov::element::Type& type,
                                           const std::vector<size_t>& filter_size,
                                           const std::vector<size_t>& strides,
                                           const std::vector<ptrdiff_t>& pads_begin,
                                           const std::vector<ptrdiff_t>& pads_end,
                                           const std::vector<size_t>& dilations,
                                           const ov::op::PadType& auto_pad,
                                           size_t num_out_channels) {
    auto shape = in.get_partial_shape();
    ov::Shape filter_weights_shape = {num_out_channels, static_cast<size_t>(shape[1].get_length())};
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());

    std::shared_ptr<ov::op::v0::Constant> filter_weights_node;
    auto tensor = ov::Tensor(type, filter_weights_shape);
    auto size = shape_size(filter_weights_shape);
    double default_value = 0.5;

    for (std::size_t i = 0; i < size; i++) {
        switch (type) {
        case ov::element::i8:
            tensor.data<ov::fundamental_type_for<ov::element::i8>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::i8>>(default_value);
            break;
        case ov::element::i16:
            tensor.data<ov::fundamental_type_for<ov::element::i16>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::i16>>(default_value);
            break;
        case ov::element::i32:
            tensor.data<ov::fundamental_type_for<ov::element::i32>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::i32>>(default_value);
            break;
        case ov::element::i64:
            tensor.data<ov::fundamental_type_for<ov::element::i64>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::i64>>(default_value);
            break;
        case ov::element::u8:
            tensor.data<ov::fundamental_type_for<ov::element::u8>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::u8>>(default_value);
            break;
        case ov::element::u16:
            tensor.data<ov::fundamental_type_for<ov::element::u16>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::u16>>(default_value);
            break;
        case ov::element::u32:
            tensor.data<ov::fundamental_type_for<ov::element::u32>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::u32>>(default_value);
            break;
        case ov::element::u64:
            tensor.data<ov::fundamental_type_for<ov::element::u64>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::u64>>(default_value);
            break;
        case ov::element::bf16:
            tensor.data<ov::fundamental_type_for<ov::element::bf16>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::bf16>>(default_value);
            break;
        case ov::element::f16:
            tensor.data<ov::fundamental_type_for<ov::element::f16>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::f16>>(default_value);
            break;
        case ov::element::f32:
            tensor.data<ov::fundamental_type_for<ov::element::f32>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::f32>>(default_value);
            break;
        case ov::element::f64:
            tensor.data<ov::fundamental_type_for<ov::element::f64>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::f64>>(default_value);
            break;
        default:
            ov::Exception::create(__FILE__,
                                  __LINE__,
                                  std::string("Not supported elment type: ") + type.get_type_name());
        }
    }

    filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);

    return std::make_shared<ov::op::v1::Convolution>(in,
                                                     filter_weights_node,
                                                     strides,
                                                     pads_begin,
                                                     pads_end,
                                                     dilations,
                                                     auto_pad);
}

std::shared_ptr<ov::Model> make_conv_pool_relu() {
    ov::Shape input_shape = {1, 1, 32, 32};
    ov::element::Type type = ov::element::f32;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});

    ov::Shape const_shape = {input_shape[0], input_shape[2], input_shape[1], input_shape[3]};
    auto const1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, const_shape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});

    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    reshape1->output(0).get_tensor().set_names({"reshape1"});

    auto conv1 = make_convolution(reshape1, type, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, ov::op::PadType::EXPLICIT, 4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});

    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ov::op::v1::MaxPool>(conv1,
                                                       stride,
                                                       padB,
                                                       padE,
                                                       kernel,
                                                       ov::op::RoundingType::FLOOR,
                                                       ov::op::PadType::EXPLICIT);
    pool1->output(0).get_tensor().set_names({"pool"});
    pool1->set_friendly_name("Pool_1");

    auto relu1 = std::make_shared<ov::op::v0::Relu>(pool1);
    relu1->set_friendly_name("Relu_1");
    relu1->output(0).get_tensor().set_names({"relu"});

    ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
    std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
    auto const2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
    const2->output(0).get_tensor().set_names({"const2"});
    const2->set_friendly_name("Const_2");

    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(relu1, const2, false);
    reshape2->output(0).get_tensor().set_names({"reshape2"});
    reshape2->set_friendly_name("Reshape_2");

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reshape2)};
    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("dummy_model");
    return model;
}

std::shared_ptr<ov::Model> make_read_concat_split_assign() {
    ov::Shape input_shape = {1, 1, 2, 4};
    ov::element::Type type = ov::element::f32;
    ov::ParameterVector parameter{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    parameter[0]->set_friendly_name("parameter");

    auto init_const = ov::op::v0::Constant::create(type, input_shape, {0});
    auto read = std::make_shared<ov::op::v3::ReadValue>(init_const, "v0");
    read->set_friendly_name("read");

    std::vector<std::shared_ptr<ov::Node>> args = {parameter[0], read};
    auto conc = std::make_shared<ov::op::v0::Concat>(args, 3);
    conc->set_friendly_name("concat");

    auto res = std::make_shared<ov::op::v0::Result>(conc);
    res->set_friendly_name("result");

    const auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
    axis->set_friendly_name("axis");

    auto crop = std::make_shared<ov::op::v1::Split>(conc, axis, 2);
    crop->set_friendly_name("split");

    auto assign = std::make_shared<ov::op::v3::Assign>(crop, "v0");
    assign->set_friendly_name("assign");

    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector{parameter});
    model->set_friendly_name("dummy_model_stateful");
    return model;
}

namespace {

const char* const BLOB_PREFIX = "blob_compatibility_";
const char* const BLOB_SUFFIX = ".blob";

}  // namespace

std::shared_ptr<ov::Model> multi_output_split_dynamic() {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto split = std::make_shared<ov::op::v1::Split>(data, axis, 2);
    auto abs = std::make_shared<ov::op::v0::Abs>(split->output(1));

    auto model = std::make_shared<ov::Model>(abs, ov::ParameterVector{data});
    model->set_friendly_name("dummy_model_dynamic_shapes");
    return model;
}

}  // namespace

namespace ov {
namespace test {
namespace behavior {

using BlobCompatibilityParams = std::tuple</* target_device = */ std::string,
                                           /* model_name = */ std::string,
                                           /* platform = */ std::string,
                                           /* ov_release = */ std::string,
                                           /* driver = */ std::string>;

class OVBlobCompatibilityNPU : public OVCompiledNetworkTestBase,
                               public testing::WithParamInterface<BlobCompatibilityParams> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::string model_name, platform, ov_release, driver;
        std::tie(target_device, model_name, platform, ov_release, driver) = this->GetParam();
        blobName = BLOB_PREFIX + model_name + "_" + platform + "_" + ov_release + "_" + driver + BLOB_SUFFIX;
        APIBaseTest::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<BlobCompatibilityParams> obj) {
        std::string target_device, model_name, platform, ov_release, driver;
        std::tie(target_device, model_name, platform, ov_release, driver) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << target_device << "_blobName=\"" << BLOB_PREFIX << model_name << "_" << platform
               << "_" << ov_release << "_" << driver << BLOB_SUFFIX << "\"";
        return result.str();
    }

protected:
    std::string blobName;
};

TEST_P(OVBlobCompatibilityNPU, CheckBlobsWithDifferentVersionsAreCompatible) {
    std::ifstream blobStream(blobName, std::ios::binary | std::ios::in);
    if (!blobStream.is_open()) {
        GTEST_FAIL() << blobName << " could not be opened. Is it located at the path of the executable?";
    }

    ov::Core core;
    OV_ASSERT_NO_THROW(core.import_model(blobStream, target_device, {ov::intel_npu::disable_version_check(true)}));
}

}  // namespace behavior

}  // namespace test

}  // namespace ov
