// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <openvino/core/partial_shape.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/type/element_type_traits.hpp"
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
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"

struct ConstRanges {
    static double max, min;
    static bool is_defined;

    static void set(double _min, double _max) {
        min = _min;
        max = _max;
        is_defined = true;
    }

    static void reset() {
        min = std::numeric_limits<double>::max();
        max = std::numeric_limits<double>::min();
        is_defined = false;
    }
};

double ConstRanges::max = std::numeric_limits<double>::max();
double ConstRanges::min = std::numeric_limits<double>::min();
bool ConstRanges::is_defined = false;

struct InputGenerateData {
    double start_from = 0;
    uint32_t range = 10;
    int32_t resolution = 1;
    int32_t seed = 1;
    bool input_attribute = false;

    InputGenerateData(double _start_from = 0,
                      uint32_t _range = 10,
                      int32_t _resolution = 1,
                      int32_t _seed = 1,
                      bool _input_attribute = false)
        : start_from(_start_from),
          range(_range),
          resolution(_resolution),
          seed(_seed),
          input_attribute(_input_attribute) {
        if (ConstRanges::is_defined) {
            auto min_orig = start_from;
            auto max_orig = start_from + range * resolution;
            auto min_ref = ConstRanges::min;
            auto max_ref = ConstRanges::max;
            if (min_orig < min_ref || min_orig == 0)
                start_from = min_ref;
            range =
                (uint32_t)round((max_orig > max_ref || max_orig == 10 ? max_ref : max_orig - start_from) - start_from);
        }
    };

    bool correct_range(const InputGenerateData new_range) {
        bool success = true;

        double new_max = new_range.start_from + new_range.range;
        double current_max = start_from + range;

        if (start_from == new_range.start_from) {
            // nothing to do - -----start_curr/new+++++++++++++++range*res curr/new-----------------------
            // nothing to do - -----start_curr/new+++++++++++++++range*res curr----------range*res new----
            // reduce range  - -----start_curr/new+++++++++++++++range*res new-----------range*res curr---
            if (current_max > new_max) {
                range = new_range.range;
                resolution = new_range.resolution > resolution ? new_range.resolution : resolution;
            }
        } else if (start_from > new_range.start_from) {
            // nothing to do        - -----start_new-----start_curr++++++++++range*res curr/new-------------------
            // nothing to do        - -----start_new-----start_curr++++++++++range*res curr------range*res new----
            // reduce range         - -----start_new-----start_curr++++++++++range*res new-------range*res curr---
            // could not find range - -----start_new---range*res new-----start_curr-----range*res curr---
            if (start_from > new_max) {
                success = false;
                // std::cout << " FAIL TO FIND RANGE: current->start_from > new_range->start_from + new_range->range "
                //           << " current->start_from: " << std::to_string(start_from)
                //           << " new_range->start_from: " << std::to_string(new_range.start_from)
                //           << " new_range max: " << std::to_string(new_max) << std::endl;
            } else if (current_max > new_max) {
                range = (uint32_t)round(new_max - start_from);
                resolution = new_range.resolution > resolution ? new_range.resolution : resolution;
            }
        } else if (start_from < new_range.start_from) {
            // reset to new         - -----start_curr-----start_new++++++++++range*res curr/new-------------------
            // reset to new         - -----start_curr-----start_new++++++++++range*res new-------range*res curr---
            // recalculate range    - -----start_curr-----start_new++++++++++range*res curr------range*res new----
            // could not find range - -----start_curr---range*res curr-----start_new-----range*res new---
            if (current_max < new_range.start_from) {
                success = false;
                // std::cout << " FAIL TO FIND RANGE: current->start_from + current->range < new_range->start_from "
                //           << " new_range start_from: " << std::to_string(new_range.start_from)
                //           << " current->start_from: " << std::to_string(start_from)
                //           << " current max: " << std::to_string(current_max) << std::endl;
            } else if (current_max >= new_max) {
                start_from = new_range.start_from;
                range = new_range.range;
                resolution = new_range.resolution > resolution ? new_range.resolution : resolution;
            } else {
                range = (uint32_t)round(current_max - new_range.start_from);
                resolution = new_range.resolution > resolution ? new_range.resolution : resolution;
                start_from = new_range.start_from;
            }
        }

        return success;
    };
};

template <class T>
void inline fill_data_random(T* pointer,
                             std::size_t size,
                             const uint32_t range = 10,
                             double_t start_from = 0,
                             const int32_t k = 1,
                             const int seed = 1) {
    if (range == 0) {
        for (std::size_t i = 0; i < size; i++) {
            pointer[i] = static_cast<T>(start_from);
        }
        return;
    }

    testing::internal::Random random(seed);
    const uint32_t k_range = k * range;  // range with respect to k
    random.Generate(k_range);

    if (start_from < 0 && !std::numeric_limits<T>::is_signed) {
        start_from = 0;
    }
    for (std::size_t i = 0; i < size; i++) {
        pointer[i] = static_cast<T>(start_from + static_cast<double>(random.Generate(k_range)) / k);
    }
}

void fill_data_boolean(ov::fundamental_type_for<ov::element::boolean>* dst, const size_t size, const int seed) {
    testing::internal::Random random(seed);
    const uint32_t range = 2;
    random.Generate(range);

    for (std::size_t i = 0; i < size; i++) {
        dst[i] = static_cast<ov::fundamental_type_for<ov::element::boolean>>(random.Generate(range));
    }
}

void fill_random_string(std::string* dst,
                        const size_t size,
                        const size_t len_range,
                        const size_t start_from,
                        const int seed) {
    static const int32_t char_range = 128;
    testing::internal::Random random_len(seed);
    random_len.Generate(len_range);
    testing::internal::Random random_char(seed);
    random_char.Generate(char_range);

    for (size_t i = 0lu; i < size; i++) {
        const auto len = start_from + static_cast<size_t>(random_len.Generate(len_range));
        auto& str = dst[i];
        str.resize(len);
        for (size_t j = 0lu; j < len; j++) {
            str[j] = static_cast<char>(random_len.Generate(char_range));
        }
    }
}

ov::Tensor create_and_fill_tensor(const ov::element::Type element_type,
                                  const ov::Shape& shape,
                                  const InputGenerateData& inGenData) {
    auto tensor = ov::Tensor(element_type, shape);
    auto size = shape_size(shape);

#define CASE(X)                                                      \
    case X:                                                          \
        fill_data_random(tensor.data<ov::fundamental_type_for<X>>(), \
                         size,                                       \
                         inGenData.range,                            \
                         inGenData.start_from,                       \
                         inGenData.resolution,                       \
                         inGenData.seed);                            \
        break;

#define CASE_CONVERT(X)                                              \
    case X: {                                                        \
        auto input = std::vector<ov::fundamental_type_for<X>>(size); \
        fill_data_random(input.data(),                               \
                         size,                                       \
                         inGenData.range,                            \
                         inGenData.start_from,                       \
                         inGenData.resolution,                       \
                         inGenData.seed);                            \
        std::memcpy(input.data(), tensor.data(), size);              \
        break;                                                       \
    }

    switch (element_type) {
        CASE(ov::element::i8)
        CASE(ov::element::i16)
        CASE(ov::element::i32)
        CASE(ov::element::i64)
        CASE(ov::element::u8)
        CASE(ov::element::u16)
        CASE(ov::element::u32)
        CASE(ov::element::u64)
        CASE(ov::element::bf16)
        CASE(ov::element::f16)
        CASE(ov::element::f32)
        CASE(ov::element::f64)
        CASE_CONVERT(ov::element::u6)
        CASE_CONVERT(ov::element::u4)
        CASE_CONVERT(ov::element::u3)
        CASE_CONVERT(ov::element::u2)
        CASE_CONVERT(ov::element::u1)
        CASE_CONVERT(ov::element::i4)
        CASE_CONVERT(ov::element::nf4)
        CASE_CONVERT(ov::element::f8e4m3)
        CASE_CONVERT(ov::element::f8e5m2)
        CASE_CONVERT(ov::element::f8e8m0)
        CASE_CONVERT(ov::element::f4e2m1)
    case ov::element::boolean:
        fill_data_boolean(static_cast<ov::fundamental_type_for<ov::element::boolean>*>(tensor.data()),
                          size,
                          inGenData.seed);
        break;
    case ov::element::Type_t::string:
        fill_random_string(static_cast<std::string*>(tensor.data()),
                           size,
                           inGenData.range,
                           inGenData.start_from,
                           inGenData.seed);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}

// Legacy impl for contrig repo
// todo: remove this after dependent repos clean up
ov::Tensor create_and_fill_tensor(const ov::element::Type element_type,
                                  const ov::Shape& shape,
                                  const uint32_t range,
                                  const double_t start_from = 0,
                                  const int32_t resolution = 1,
                                  const int seed = 1) {
    return create_and_fill_tensor(element_type, shape, InputGenerateData(start_from, range, resolution, seed));
}

std::shared_ptr<ov::Node> make_convolution(const ov::Output<ov::Node>& in,
                                           const ov::element::Type& type,
                                           const std::vector<size_t>& filter_size,
                                           const std::vector<size_t>& strides,
                                           const std::vector<ptrdiff_t>& pads_begin,
                                           const std::vector<ptrdiff_t>& pads_end,
                                           const std::vector<size_t>& dilations,
                                           const ov::op::PadType& auto_pad,
                                           size_t num_out_channels,
                                           bool add_biases = false,
                                           const std::vector<float>& filter_weights = {},
                                           const std::vector<float>& biases_weights = {}) {
    auto shape = in.get_partial_shape();
    ov::Shape filter_weights_shape = {num_out_channels, static_cast<size_t>(shape[1].get_length())};
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());

    std::shared_ptr<ov::op::v0::Constant> filter_weights_node;
    if (!filter_weights.empty()) {
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(type, filter_weights_shape, filter_weights);
    } else {
        auto tensor = create_and_fill_tensor(type, filter_weights_shape, 9, 1);
        filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    auto conv = std::make_shared<ov::op::v1::Convolution>(in,
                                                          filter_weights_node,
                                                          strides,
                                                          pads_begin,
                                                          pads_end,
                                                          dilations,
                                                          auto_pad);
    if (add_biases) {
        std::shared_ptr<ov::op::v0::Constant> biases_weights_node;
        const size_t rank = in.get_partial_shape().rank().get_length();
        ov::Shape bias_shape(rank, 1);
        bias_shape[1] = num_out_channels;
        if (!biases_weights.empty()) {
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(type, bias_shape, biases_weights);
        } else {
            auto tensor = create_and_fill_tensor(type, bias_shape, 9, 1);
            biases_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto add = std::make_shared<ov::op::v1::Add>(conv, biases_weights_node);
        return add;
    } else {
        return conv;
    }
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
    return model;
}

std::shared_ptr<ov::Model> multi_output_split_dynamic() {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto split = std::make_shared<ov::op::v1::Split>(data, axis, 2);
    auto abs = std::make_shared<ov::op::v0::Abs>(split->output(1));

    return std::make_shared<ov::Model>(abs, ov::ParameterVector{data});
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
    model->set_friendly_name("ReadConcatSplitAssign");
    return model;
}

int main(int argc, char* argv[]) {
    ov::Core core;
    const char* DEVICE_NPU = "NPU";
    auto driver_version = core.get_property(DEVICE_NPU, ov::intel_npu::driver_version.name()).as<std::string>();
    auto platform = core.get_property(DEVICE_NPU, "NPU_PLATFORM").as<std::string>();
    auto version = ov::get_openvino_version().buildNumber;

    std::cout << "OpenVINO version: " << version << std::endl;
    std::cout << "NPU platform: " << platform << std::endl;
    std::cout << "Driver Version: " << driver_version << std::endl;

    auto models = std::vector<std::function<std::shared_ptr<ov::Model>(void)>>{&make_conv_pool_relu,
                                                                               &multi_output_split_dynamic,
                                                                               &make_read_concat_split_assign};

    return 0;
}
