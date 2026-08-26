// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <tuple>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

namespace {
std::tuple<std::shared_ptr<ov::Model>, size_t> makeModelWithWeights(size_t elem_count) {
    size_t weight_count = 0;
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{elem_count});
    auto one = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{elem_count}, {1});
    weight_count++;
    auto add = std::make_shared<ov::op::v1::Add>(s, one);
    auto three = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{elem_count}, {3});
    weight_count++;
    auto mul = std::make_shared<ov::op::v1::Multiply>(add, three);
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    auto m = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{s}, "producer");
    m->input(0).set_names({"input"});
    m->output(0).set_names({"output"});
    return std::make_tuple(std::move(m), weight_count);
}

std::vector<std::shared_ptr<ov::op::v0::Constant>> collectConstants(ov::Model& model) {
    std::vector<std::shared_ptr<ov::op::v0::Constant>> constant_to_share;
    for (const auto& op : model.get_ops()) {
        auto shared_weight_candidate = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
        if (!shared_weight_candidate) {
            continue;
        }
        constant_to_share.push_back(shared_weight_candidate);
    }
    return constant_to_share;
}

std::tuple<std::shared_ptr<ov::weight_sharing::Context>, std::shared_ptr<::ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>> makeWeightSharingContext(
    std::vector<std::shared_ptr<ov::op::v0::Constant>>&& constants,
    size_t alignment,
    std::function<size_t()> source_id_generator,
    size_t skew_constant_alignment = 0) {
    // Allocate a single shared buffer for all constants, with padding to the specified alignment.
    size_t source_buffer_size = 0;
    for (const auto& constant : constants) {
        source_buffer_size += ov::util::align_size_up(constant->get_byte_size(), alignment) + skew_constant_alignment;
    }

    const size_t source_id = source_id_generator();
    auto raw = std::make_shared<ov::AlignedBuffer>(source_buffer_size, alignment);

    auto source_buffer = std::make_shared<::ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(raw->get_ptr<char>(),
                                                                                                  raw->size(),
                                                                                                  raw,
                                                                                                  ::ov::create_base_descriptor(source_id, 0, raw));

    // slate the source buffer to hold all constants inside it with offets equal to the requested alignment.
    size_t constant_id = skew_constant_alignment;  // constants ID is a weight offset in the shared source buffer
    std::vector<std::shared_ptr<ov::op::v0::Constant>> shared_constants;
    auto shared_ctx_ptr = std::make_shared<ov::weight_sharing::Context>();
    ov::weight_sharing::set_weight_source(*shared_ctx_ptr, source_buffer);
    for (const auto& constant : constants) {
        auto const_descriptor = ::ov::create_base_descriptor(source_id, constant_id, source_buffer);
        auto constant_shared_buffer = std::make_shared<::ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(source_buffer->get_ptr<char>() + constant_id,
                                                                                                               constant->get_byte_size(),
                                                                                                               source_buffer,
                                                                                                               const_descriptor);
        constant_id += ov::util::align_size_up(constant->get_byte_size(), alignment) + skew_constant_alignment;
        auto shared_constant = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(), constant->get_shape(), constant_shared_buffer);
        shared_constant->set_friendly_name(constant->get_friendly_name());
        std::memcpy(constant_shared_buffer->get_ptr(), constant->get_data_ptr(), constant->get_byte_size());

        ov::replace_node(constant, shared_constant);

        // register the shared constant in the weight sharing context
        ov::weight_sharing::set_constant(*shared_ctx_ptr, *shared_constant);
        ov::weight_sharing::set_runtime_weight_source(*shared_ctx_ptr, source_buffer);
    }
    return std::make_tuple(std::move(shared_ctx_ptr), std::move(source_buffer));
}

std::tuple<std::shared_ptr<ov::weight_sharing::Context>, std::shared_ptr<::ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>>
makeWeightSharingContextIncorrectAlignment(std::vector<std::shared_ptr<ov::op::v0::Constant>>&& constants, std::function<size_t()> source_id_generator) {
    size_t alignment_not_equal_to_page_size = 64;
    // This skew will cause the constants to be misaligned in the shared buffer, which should trigger an error during
    // during GPU memory importing.
    // The skew is required to be non-zero, as the system memory manager call allocates memory aligned to the system page size,
    // even if we request 64 bytes-alignment
    size_t individual_constant_alignment_skew = 13;
    return makeWeightSharingContext(std::move(constants), alignment_not_equal_to_page_size, source_id_generator, individual_constant_alignment_skew);
}

std::tuple<std::shared_ptr<ov::weight_sharing::Context>, std::shared_ptr<::ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>>
makeCorrectWeightSharingContext(std::vector<std::shared_ptr<ov::op::v0::Constant>>&& constants, std::function<size_t()> source_id_generator) {
    size_t page_alignment = static_cast<size_t>(ov::util::get_system_page_size());
    return makeWeightSharingContext(std::move(constants), page_alignment, source_id_generator, 0);
}

class WeightContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto cache_file_path = std::filesystem::path(testing::UnitTest::GetInstance()->current_test_info()->name()).replace_extension(".bin");
        if (std::filesystem::exists(cache_file_path)) {
            std::filesystem::remove(cache_file_path);
        }
        properties = ov::AnyMap{{ov::cache_path(cache_file_path)}};
    }

    void TearDown() override {
        auto cache_path_it = properties.find(ov::cache_path.name());
        assert(cache_path_it != properties.end());
        auto cache_path = cache_path_it->second.as<std::filesystem::path>();
        std::filesystem::remove(cache_path);
    }

    void enableWeightSharingContext(const ov::internal::WeightSharingCtxPtr& weightCtx) {
        const ov::internal::WeightSharingCtxPtr shared_ctx = weightCtx;
        properties[ov::internal::model_sharing_context.name()] = ov::Any{shared_ctx};
    }

    const ov::AnyMap& getProperties() const {
        return properties;
    }

private:
    ov::AnyMap properties;
};
}  // namespace

TEST_F(WeightContextTest, smoke_weightContextCannotBeConsumedDueToUnsupportedAlignment) {
    auto core = ov::Core();
    const size_t weight_size = 1024;
    auto [model, weight_count] = makeModelWithWeights(weight_size);
    auto shared_constant_candidates = collectConstants(*model);
    ASSERT_EQ(shared_constant_candidates.size(), weight_count);

    auto [weightCtx, sourceBuffer] = makeWeightSharingContextIncorrectAlignment(std::move(shared_constant_candidates), []() {
        return 1;
    });
    ASSERT_TRUE(sourceBuffer->size() >= weight_size * weight_count);

    enableWeightSharingContext(weightCtx);
    EXPECT_THROW(core.compile_model(model, "GPU", getProperties()), ov::Exception);
}

TEST_F(WeightContextTest, smoke_weightContextInferenceWithSharedWeights) {
    auto core = ov::Core();
    const size_t weight_size = 1024;
    auto [model_to_mutate, weight_count] = makeModelWithWeights(weight_size);
    auto shared_constant_candidates = collectConstants(*model_to_mutate);
    ASSERT_EQ(shared_constant_candidates.size(), weight_count);

    auto [weightCtx, sourceBuffer] = makeCorrectWeightSharingContext(std::move(shared_constant_candidates), []() {
        return 2;
    });
    enableWeightSharingContext(weightCtx);
    auto mutated_exec_net = core.compile_model(model_to_mutate, "GPU", getProperties());
    auto mutated_inf_req = mutated_exec_net.create_infer_request();

    auto [intact_model, intact_weight_count] = makeModelWithWeights(weight_size);
    auto intact_exec_net = core.compile_model(intact_model, "GPU");
    auto intact_inf_req = intact_exec_net.create_infer_request();

    ov::Tensor input_tensor(ov::element::u8, ov::Shape{weight_size});
    OV_ASSERT_NO_THROW(mutated_inf_req.set_input_tensor(input_tensor));
    OV_ASSERT_NO_THROW(mutated_inf_req.infer());
    OV_ASSERT_NO_THROW(intact_inf_req.set_input_tensor(input_tensor));
    OV_ASSERT_NO_THROW(intact_inf_req.infer());
    ASSERT_EQ(
        std::memcmp(intact_inf_req.get_output_tensor().data(), mutated_inf_req.get_output_tensor().data(), intact_inf_req.get_output_tensor().get_byte_size()),
        0);
}
