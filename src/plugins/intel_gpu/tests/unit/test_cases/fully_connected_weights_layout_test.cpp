// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <sstream>
#include <string>

#include "fully_connected_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/op/fc_weights_layout.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "registry/implementation_manager.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;
using ov::intel_gpu::op::FcWeightsLayout;

namespace {

std::shared_ptr<ov::Node> weights(const ov::Shape& shape, ov::element::Type et = ov::element::f16) {
    return ov::op::v0::Constant::create(et, shape, std::vector<float>(ov::shape_size(shape), 1.0f));
}

std::shared_ptr<ov::Node> activations(const ov::PartialShape& shape) {
    return std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
}

std::shared_ptr<ov::intel_gpu::op::FullyConnected> make_fc(const ov::PartialShape& a,
                                                           const ov::Shape& w,
                                                           bool transpose_b,
                                                           std::optional<FcWeightsLayout> layout = std::nullopt) {
    return std::make_shared<ov::intel_gpu::op::FullyConnected>(activations(a),
                                                               weights(w),
                                                               std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                                               ov::element::f16,
                                                               transpose_b,
                                                               layout);
}

// Records string / bool attributes, or writes them back from the recorded values.
class TestVisitor : public ov::AttributeVisitor {
public:
    explicit TestVisitor(bool load) : m_load(load) {}
    std::map<std::string, std::string> strings;
    std::map<std::string, bool> bools;

    using ov::AttributeVisitor::on_adapter;
    void on_adapter(const std::string&, ov::ValueAccessor<void>&) override {}
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        if (!m_load)
            strings[name] = adapter.get();
        else if (strings.count(name))
            adapter.set(strings.at(name));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        if (!m_load)
            bools[name] = adapter.get();
        else if (bools.count(name))
            adapter.set(bools.at(name));
    }

private:
    bool m_load;
};

}  // namespace

TEST(fc_weights_layout, weight_dims) {
    using ov::intel_gpu::op::fc_weight_dims;
    const auto nk = fc_weight_dims(ov::PartialShape{32, 128}, FcWeightsLayout::NK);
    EXPECT_EQ(nk.N, 32);
    EXPECT_EQ(nk.K, 128);
    const auto kn = fc_weight_dims(ov::PartialShape{128, 32}, FcWeightsLayout::KN);
    EXPECT_EQ(kn.N, 32);
    EXPECT_EQ(kn.K, 128);
    const auto gnk = fc_weight_dims(ov::PartialShape{2, 32, 64}, FcWeightsLayout::GNK);
    EXPECT_EQ(gnk.N, 32);
    EXPECT_EQ(gnk.K, 128);
    EXPECT_EQ(gnk.G, 2);
    EXPECT_EQ(gnk.gs, 64);
    EXPECT_THROW(fc_weight_dims(ov::PartialShape{32, 128}, FcWeightsLayout::GNK), ov::Exception);
}

TEST(fc_weights_layout, string_round_trip) {
    for (auto l : {FcWeightsLayout::NK, FcWeightsLayout::KN, FcWeightsLayout::GNK})
        EXPECT_EQ(ov::intel_gpu::op::fc_weights_layout_from_string(ov::intel_gpu::op::to_string(l)), l);
    EXPECT_THROW(ov::intel_gpu::op::fc_weights_layout_from_string("NKG"), ov::Exception);
}

TEST(fc_weights_layout, op_shape_inference) {
    auto nk = make_fc({4, 128}, {32, 128}, true);
    EXPECT_EQ(nk->get_weights_layout(), FcWeightsLayout::NK);
    EXPECT_EQ(nk->get_output_partial_shape(0), (ov::PartialShape{4, 32}));

    auto kn = make_fc({4, 128}, {128, 32}, false);
    EXPECT_EQ(kn->get_weights_layout(), FcWeightsLayout::KN);
    EXPECT_EQ(kn->get_output_partial_shape(0), (ov::PartialShape{4, 32}));

    auto gnk = make_fc({-1, 128}, {2, 32, 64}, true, FcWeightsLayout::GNK);
    EXPECT_EQ(gnk->get_weights_layout(), FcWeightsLayout::GNK);
    EXPECT_EQ(gnk->get_output_partial_shape(0), (ov::PartialShape{-1, 32}));

    auto cloned = gnk->clone_with_new_inputs(gnk->input_values());
    EXPECT_EQ(ov::as_type_ptr<ov::intel_gpu::op::FullyConnected>(cloned)->get_weights_layout(), FcWeightsLayout::GNK);
}

TEST(fc_weights_layout, op_rejects_inconsistent_layout) {
    EXPECT_THROW(make_fc({4, 128}, {2, 32, 64}, false, FcWeightsLayout::GNK), ov::Exception);
    EXPECT_THROW(make_fc({4, 128}, {32, 128}, true, FcWeightsLayout::KN), ov::Exception);
    EXPECT_THROW(make_fc({4, 128}, {128, 32}, false, FcWeightsLayout::NK), ov::Exception);
}

TEST(fc_weights_layout, compressed_op_shape_inference) {
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(activations({4, 128}),
                                                                            weights({2, 32, 64}, ov::element::u8),
                                                                            std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                                                            weights({2, 32}),
                                                                            weights({2, 32}, ov::element::u8),
                                                                            ov::element::f16,
                                                                            true,
                                                                            false,
                                                                            false,
                                                                            FcWeightsLayout::GNK);
    EXPECT_EQ(fc->get_output_partial_shape(0), (ov::PartialShape{4, 32}));
    auto cloned = ov::as_type_ptr<ov::intel_gpu::op::FullyConnectedCompressed>(fc->clone_with_new_inputs(fc->input_values()));
    EXPECT_EQ(cloned->get_weights_layout(), FcWeightsLayout::GNK);
    EXPECT_FALSE(cloned->get_transpose_b_scale());
}

TEST(fc_weights_layout, op_visit_attributes_round_trip) {
    auto gnk = make_fc({4, 128}, {2, 32, 64}, true, FcWeightsLayout::GNK);
    TestVisitor save(false);
    gnk->visit_attributes(save);
    EXPECT_EQ(save.strings.at("weights_layout"), "GNK");

    TestVisitor load(true);
    load.strings = save.strings;
    load.bools = save.bools;
    ov::intel_gpu::op::FullyConnected restored;
    restored.visit_attributes(load);
    EXPECT_EQ(restored.get_weights_layout(), FcWeightsLayout::GNK);

    // An IR without the attribute falls back to NK / KN from transpose_b.
    TestVisitor legacy(true);
    legacy.bools["transpose_b"] = false;
    ov::intel_gpu::op::FullyConnected legacy_fc;
    legacy_fc.visit_attributes(legacy);
    EXPECT_EQ(legacy_fc.get_weights_layout(), FcWeightsLayout::KN);
}

TEST(fc_weights_layout, primitive_default_hash_and_serialization) {
    EXPECT_EQ(fully_connected("fc", input_info("in"), "w", "", 2, 2, true).weights_layout, FcWeightsLayout::NK);
    EXPECT_EQ(fully_connected("fc", input_info("in"), "w", "", 2, 2, false).weights_layout, FcWeightsLayout::KN);

    const fully_connected nk("fc", input_info("in"), "w", "", data_types::f16, 2, 3, true);
    const fully_connected gnk("fc", input_info("in"), "w", "", data_types::f16, 2, 3, true, FcWeightsLayout::GNK);
    EXPECT_NE(nk.hash(), gnk.hash());
    EXPECT_FALSE(nk == gnk);

    auto& engine = get_test_engine();
    std::stringstream storage;
    BinaryOutputBuffer ob(storage);
    gnk.save(ob);
    BinaryInputBuffer ib(storage, engine);
    fully_connected loaded;
    loaded.load(ib);
    EXPECT_EQ(loaded.weights_layout, FcWeightsLayout::GNK);
    EXPECT_EQ(loaded.hash(), gnk.hash());
}

TEST(fc_weights_layout, primitive_rejects_inconsistent_layout) {
    EXPECT_THROW(fully_connected("fc", input_info("in"), "w", "", 2, 2, true, FcWeightsLayout::KN), ov::Exception);
    EXPECT_THROW(fully_connected("fc", input_info("in"), "w", "", 2, 2, false, FcWeightsLayout::NK), ov::Exception);
    EXPECT_THROW(fully_connected("fc", input_info("in"), "w", "", 2, 3, false, FcWeightsLayout::GNK), ov::Exception);
    EXPECT_THROW(fully_connected("fc", input_info("in"), "w", "", data_types::f16, 2, 2, true, FcWeightsLayout::KN), ov::Exception);
    EXPECT_THROW(fully_connected("fc", input_info("in"), "w", "", data_types::f16, 2, 3, false, FcWeightsLayout::GNK), ov::Exception);
}

TEST(fc_weights_layout, gnk_legacy_shape_inference) {
    auto& engine = get_test_engine();
    const int64_t G = 2, N = 32, gs = 64, K = G * gs;
    topology topo(input_layout("input", layout{ov::PartialShape{4, K}, data_types::f16, format::bfyx}));
    // Legacy constant lowering pads [G, N, gs] to bfyx [G, N, gs, 1].
    topo.add(data("weights", engine.allocate_memory({ov::PartialShape{G, N, gs, 1}, data_types::f16, format::bfyx})));
    topo.add(fully_connected("fc", input_info("input"), "weights", "", data_types::f16, 2, 3, true, FcWeightsLayout::GNK));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(false));
    auto prog = program::build_program(engine, topo, config, false, true);
    auto& fc = prog->get_node("fc").as<fully_connected>();
    // Legacy shape inference yields a 4D bfyx output.
    const auto out = fc.get_output_layout();
    EXPECT_EQ(out.batch(), 4);
    EXPECT_EQ(out.feature(), N);
}

TEST(fc_weights_layout, gnk_shape_inference_and_impl_rejection) {
    auto& engine = get_test_engine();
    const int64_t G = 2, N = 32, gs = 64, K = G * gs;
    topology topo(input_layout("input", layout{ov::PartialShape{4, K}, data_types::f16, format::bfyx}));
    topo.add(data("weights", engine.allocate_memory({ov::PartialShape{G, N, gs}, data_types::f16, format::bfyx})));
    topo.add(fully_connected("fc", input_info("input"), "weights", "", data_types::f16, 2, 3, true, FcWeightsLayout::GNK));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topo, config, false, true);
    auto& fc = prog->get_node("fc").as<fully_connected>();
    EXPECT_EQ(fc.get_output_layout().get_partial_shape(), (ov::PartialShape{4, N}));

    // No OCL / oneDNN implementation may accept GNK weights.
    for (const auto& impl : fc.type()->get_supported_implementations(fc)) {
        EXPECT_NE(impl->get_impl_type(), impl_types::ocl);
        EXPECT_NE(impl->get_impl_type(), impl_types::onednn);
    }
}
