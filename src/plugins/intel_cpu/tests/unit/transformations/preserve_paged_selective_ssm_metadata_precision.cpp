// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/preserve_paged_selective_ssm_metadata_precision.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "nodes/paged_selective_ssm_ports.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/cpu_opset/common/pass/insert_convert_after_extension.hpp"

namespace ov::intel_cpu::test {
namespace {

using MetadataInputs = std::array<ov::Output<ov::Node>, paged_ssm_metadata_ports.size()>;

struct PagedGraph {
    std::shared_ptr<ov::op::internal::PagedSelectiveSSM> op;
    ov::ParameterVector computation_parameters;
};

std::shared_ptr<ov::op::util::FrameworkNode> make_i64_metadata_extension() {
    auto extension = std::make_shared<ov::op::util::FrameworkNode>(ov::OutputVector{});
    extension->set_output_type(0, ov::element::i64, ov::PartialShape{-1});
    extension->cache_output_descriptor();
    return extension;
}

PagedGraph make_paged_graph(const MetadataInputs& metadata) {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 4});
    auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 16});
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 4, 8});
    auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 16});
    auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 4, 8, 16});

    ov::OutputVector inputs{A, dt, B, x, C, state};
    inputs.insert(inputs.end(), metadata.begin(), metadata.end());
    return {std::make_shared<ov::op::internal::PagedSelectiveSSM>(inputs), {A, dt, B, x, C, state}};
}

void run_cpu_i64_conversion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<PreservePagedSelectiveSSMMetadataPrecision>();
    manager.register_pass<ov::pass::InsertConvertAfterExtension>(false);
    manager.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::i64, ov::element::i32}},
                                                      type_to_fuse_map{},
                                                      false,
                                                      false);
    manager.run_passes(model);
}

TEST(PreservePagedSelectiveSSMMetadataPrecisionTest, KeepsUnknownExtensionOutputI64) {
    auto metadata = make_i64_metadata_extension();
    MetadataInputs metadata_inputs;
    metadata_inputs.fill(metadata);
    auto graph = make_paged_graph(metadata_inputs);
    auto model = std::make_shared<ov::Model>(graph.op->outputs(), graph.computation_parameters);

    run_cpu_i64_conversion(model);

    for (const auto port : paged_ssm_metadata_ports) {
        const auto input = graph.op->input_value(input_port_index(port));
        EXPECT_EQ(input.get_element_type(), ov::element::i64);
        EXPECT_EQ(input.get_node_shared_ptr(), metadata);
    }
}

TEST(PreservePagedSelectiveSSMMetadataPrecisionTest, KeepsAllMetadataInputsI64) {
    auto metadata = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    MetadataInputs metadata_inputs;
    metadata_inputs.fill(metadata);
    auto graph = make_paged_graph(metadata_inputs);
    auto parameters = graph.computation_parameters;
    parameters.push_back(metadata);
    auto model = std::make_shared<ov::Model>(graph.op->outputs(), parameters);

    run_cpu_i64_conversion(model);

    for (const auto port : paged_ssm_metadata_ports) {
        EXPECT_EQ(graph.op->get_input_element_type(input_port_index(port)), ov::element::i64);
    }
}

TEST(PreservePagedSelectiveSSMMetadataPrecisionTest, KeepsDerivedLargeValuesExact) {
    constexpr int64_t large_value = int64_t{1} << 32;
    auto metadata = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    auto large_values = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {large_value, large_value + 1});
    auto zeros = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {int64_t{0}, int64_t{0}});
    auto processed_tokens = std::make_shared<ov::op::v1::Add>(large_values, zeros);
    MetadataInputs metadata_inputs{metadata, metadata, metadata, processed_tokens, metadata};
    auto graph = make_paged_graph(metadata_inputs);
    auto parameters = graph.computation_parameters;
    parameters.push_back(metadata);
    auto model = std::make_shared<ov::Model>(graph.op->outputs(), parameters);

    run_cpu_i64_conversion(model);

    const auto processed_port = input_port_index(PagedSelectiveSSMInputPort::NumProcessedTokens);
    EXPECT_EQ(graph.op->get_input_element_type(processed_port), ov::element::i64);
    const auto folded = ov::util::get_constant_from_source(graph.op->input_value(processed_port));
    ASSERT_NE(folded, nullptr);
    EXPECT_EQ(folded->cast_vector<int64_t>(), (std::vector<int64_t>{large_value, large_value + 1}));
}

TEST(PreservePagedSelectiveSSMMetadataPrecisionTest, ConvertsUnrelatedSharedConsumerToI32) {
    constexpr size_t shared_input_port = 0;
    auto metadata = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    MetadataInputs metadata_inputs;
    metadata_inputs.fill(metadata);
    auto graph = make_paged_graph(metadata_inputs);
    auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {int64_t{0}});
    auto unrelated = std::make_shared<ov::op::v1::Add>(metadata, zero);
    auto parameters = graph.computation_parameters;
    parameters.push_back(metadata);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{graph.op, unrelated}, parameters);

    run_cpu_i64_conversion(model);

    EXPECT_EQ(graph.op->get_input_element_type(input_port_index(PagedSelectiveSSMInputPort::SubsequenceBegins)),
              ov::element::i64);
    EXPECT_EQ(unrelated->get_input_element_type(shared_input_port), ov::element::i32);
    const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(unrelated->get_input_node_shared_ptr(shared_input_port));
    ASSERT_NE(convert, nullptr);
    EXPECT_EQ(convert->input_value(0).get_node_shared_ptr(), metadata);
}

TEST(PreservePagedSelectiveSSMMetadataPrecisionTest, LeavesI32MetadataUnchanged) {
    auto metadata = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    MetadataInputs metadata_inputs;
    metadata_inputs.fill(metadata);
    auto graph = make_paged_graph(metadata_inputs);
    auto parameters = graph.computation_parameters;
    parameters.push_back(metadata);
    auto model = std::make_shared<ov::Model>(graph.op->outputs(), parameters);

    PreservePagedSelectiveSSMMetadataPrecision pass;
    EXPECT_FALSE(pass.run_on_model(model));
    for (const auto port : paged_ssm_metadata_ports) {
        EXPECT_EQ(graph.op->get_input_element_type(input_port_index(port)), ov::element::i32);
    }
}

}  // namespace
}  // namespace ov::intel_cpu::test
