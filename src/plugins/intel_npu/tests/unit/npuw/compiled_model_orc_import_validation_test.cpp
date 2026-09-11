// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "compiled_model.hpp"
#include "attn/attn_subgraph.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "moe/moe_subgraph.hpp"
#include "openvino/opsets/opset10.hpp"
#include "orc.hpp"
#include "orc/schema_npuw.hpp"
#include "serialization.hpp"
#include "weights_bank.hpp"

namespace {

namespace orc = ov::npuw::orc;

using CM = ov::npuw::CompiledModel;
using ToSubmodel = CM::ToSubmodel;

class NullPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::compile_model call");
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::compile_model(context) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(stream) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(stream, context) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(blob) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(blob, context) call");
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::query_model call");
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        return {};
    }
};

// The single global input / global output the forged blob advertises.
std::shared_ptr<const ov::Model> make_meta_model() {
    auto param = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1});
    param->output(0).get_tensor().set_names({"input"});
    auto relu = std::make_shared<ov::opset10::Relu>(param);
    auto result = std::make_shared<ov::opset10::Result>(relu);
    result->output(0).get_tensor().set_names({"output"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "RoutingValidationModel");
}

struct Tables {
    std::vector<ToSubmodel> inputs;
    std::vector<ToSubmodel> outputs;
    std::map<std::size_t, std::vector<ToSubmodel>> param_subscribers;
    std::map<ToSubmodel, ToSubmodel> links;
};

// A well-formed single-submodel routing: global input 0 -> sub0/in0,
// sub0/out0 -> global output 0, plus a self-consistent interconnect link.
Tables make_valid_tables() {
    Tables t;
    t.inputs = {ToSubmodel{0, 0}};
    t.outputs = {ToSubmodel{0, 0}};
    t.links = {{ToSubmodel{0, 1}, ToSubmodel{0, 0}}};
    return t;
}

// Mirrors CompiledModelDesc::serialize() for a submodel that carries no compiled
// blob of its own - the only shape a test can produce without a real NPU device.
// `closure_uid` also fixes the closure/is_remote/closure_size triple: a non-empty
// vector produces a bank-backed closure entry that reconstruct_closure() resolves
// through submodel_device(replaced_by) - the unchecked sink this file guards against.
void write_submodel(std::ostream& buffer,
                    std::optional<std::size_t> replaced_by,
                    std::vector<int64_t> closure_uid = {}) {
    orc::with_leaf_section(buffer, static_cast<orc::TypeId>(orc::schema_npuw::Subgraph::ID), 0u, [&] {
        auto stream = orc::Stream::writer(buffer);

        std::size_t device_index = 0u;
        bool has_compiled_model = false;
        stream & device_index & has_compiled_model;

        std::size_t param_base = 0u;
        bool forced_to_fcall = false;
        int64_t gather_dst = -1, gather_src = -1, gather_idx = -1;
        int64_t quant_dst = -1, quant_w = -1, quant_z = -1, quant_s = -1, quant_idx = -1;
        std::optional<ov::npuw::compiled::Spatial> spatial;
        stream & replaced_by & param_base & forced_to_fcall & gather_dst & gather_src & gather_idx & quant_dst &
            quant_w & quant_z & quant_s & quant_idx & spatial;

        // Mirrors CompiledModelDesc::serialize()'s is_fcall gate: a genuine call site
        // (replaced_by set, no compiled_model of its own) skips this section; anything
        // else - including this harness's stand-in for an ordinary submodel - must write
        // it, with an empty context standing in for "no MoE/attention state".
        if (!replaced_by.has_value()) {
            ov::npuw::v1::subgraphs::Context empty_context;
            ov::npuw::moe::serialize_compiled_state(empty_context, stream, nullptr);
            ov::npuw::attn::serialize_compiled_state(empty_context, stream, nullptr);
        }

        std::vector<bool> is_remote(closure_uid.size(), false);
        stream & is_remote & closure_uid;

        std::vector<ov::Tensor> scales, zerops;
        std::size_t closure_size = closure_uid.size();
        std::vector<std::size_t> cpu_closure_ids;  // none of the closure entries are host-resident
        stream & scales & zerops & closure_size & cpu_closure_ids;
    });
}

// One forged Subgraph section: an optional function-body reference plus an
// optional bank-backed closure entry (see write_submodel()).
struct SubmodelSpec {
    std::optional<std::size_t> replaced_by;
    std::vector<int64_t> closure_uid;
};

// Produces exactly what CompiledModel::export_model() would, except the routing
// tables and the submodels are whatever the caller asks for.
std::string make_blob(const Tables& tables, const std::vector<SubmodelSpec>& submodels) {
    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    orc::write_file_header(buffer, orc::schema_npuw::NPUW_ORC_PARTITIONED_SCHEMA);

    orc::with_section(buffer, CM::kOrcType, CM::kOrcVersion, 0u, [&] {
        orc::with_leaf_section(buffer, orc::META_SECTION_TYPE, 0u, [&] {
            auto stream = orc::Stream::writer(buffer);

            std::string model_name = "RoutingValidationModel";
            const auto model = make_meta_model();
            auto model_inputs = model->inputs();
            auto model_outputs = model->outputs();
            stream & model_name & model_inputs & model_outputs;

            auto copy = tables;
            stream & copy.inputs & copy.outputs & copy.param_subscribers & copy.links;

            std::vector<std::string> dev_list{"NPU"};
            stream & dev_list;

            auto options = std::make_shared<::intel_npu::OptionsDesc>();
            ::intel_npu::registerNPUWOptions(*options);
            ::intel_npu::Config cfg(options);
            // Config::fromString() cannot parse an empty string, and a real blob never carries one
            cfg.update({{std::string(::intel_npu::NPUW_DEVICES::key()), "NPU"}});
            stream & cfg;

            ov::AnyMap props;
            stream & props;

            bool is_weightless = false;
            stream & is_weightless;

            ov::npuw::s11n::BF16Cache bf16_consts;
            stream & bf16_consts;
        });

        for (const auto& submodel : submodels) {
            write_submodel(buffer, submodel.replaced_by, submodel.closure_uid);
        }

        orc::with_leaf_section(buffer, ov::npuw::weights::Bank::kOrcType, ov::npuw::weights::Bank::kOrcVersion, [&] {
            auto stream = orc::Stream::writer(buffer);
            std::string bank_name = "routing_validation_bank";
            std::size_t bank_size = 0u;
            stream & bank_name & bank_size;
        });
    });

    return buffer.str();
}

// A single ordinary (non-function) submodel - the shape every test below uses unless
// it needs to forge the function-body reference itself. replaced_by is unset, matching
// a real ordinary submodel.
const std::vector<SubmodelSpec> kOneSubmodel{SubmodelSpec{std::nullopt, {}}};

void import_blob(const Tables& tables, const std::vector<SubmodelSpec>& submodels = kOneSubmodel) {
    const auto plugin = std::make_shared<NullPlugin>();
    auto bytes = make_blob(tables, submodels);
    std::stringstream stream(bytes, std::ios::in | std::ios::out | std::ios::binary);
    CM::import_model(stream, plugin, {});
}

TEST(NpuwImportRoutingValidation, WellFormedBlobImports) {
    EXPECT_NO_THROW(import_blob(make_valid_tables()));
}

// The reported issue: the blob declares a link into a submodel index far beyond the
// number of Subgraph sections it actually carries. MemAccessSim used to index its
// m_read_list vector with it, yielding an out-of-bounds write.
TEST(NpuwImportRoutingValidation, LinkConsumerSubmodelIndexOutOfRange) {
    auto tables = make_valid_tables();
    tables.links = {{ToSubmodel{0x100000, 0}, ToSubmodel{0, 0}}};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "submodel input link refers to submodel 1048576 while only 1 submodel(s) are present");
}

TEST(NpuwImportRoutingValidation, LinkProducerSubmodelIndexOutOfRange) {
    auto tables = make_valid_tables();
    tables.links = {{ToSubmodel{0, 0}, ToSubmodel{0x100000, 0}}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        import_blob(tables),
        ov::Exception,
        "submodel output link refers to submodel 1048576 while only 1 submodel(s) are present");
}

TEST(NpuwImportRoutingValidation, GlobalInputMappingOutOfRange) {
    auto tables = make_valid_tables();
    tables.inputs = {ToSubmodel{9, 0}};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "global input mapping refers to submodel 9 while only 1 submodel(s) are present");
}

TEST(NpuwImportRoutingValidation, GlobalOutputMappingOutOfRange) {
    auto tables = make_valid_tables();
    tables.outputs = {ToSubmodel{9, 0}};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "global output mapping refers to submodel 9 while only 1 submodel(s) are present");
}

// init_gio() indexes both vectors by the model's real input()/output() count, not by
// the vector's own size - a short vector reaches an unchecked .at() there, and
// report_io() indexes an oversized one straight into the model's port array.
TEST(NpuwImportRoutingValidation, GlobalInputMappingWrongCardinality) {
    auto tables = make_valid_tables();
    tables.inputs = {};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "global input mapping has 0 entries but the model has 1 input(s)");

    tables = make_valid_tables();
    tables.inputs = {ToSubmodel{0, 0}, ToSubmodel{0, 0}};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "global input mapping has 2 entries but the model has 1 input(s)");
}

TEST(NpuwImportRoutingValidation, GlobalOutputMappingWrongCardinality) {
    auto tables = make_valid_tables();
    tables.outputs = {};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "global output mapping has 0 entries but the model has 1 output(s)");

    tables = make_valid_tables();
    tables.outputs = {ToSubmodel{0, 0}, ToSubmodel{0, 0}};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "global output mapping has 2 entries but the model has 1 output(s)");
}

TEST(NpuwImportRoutingValidation, ParamSubscriberOutOfRange) {
    auto tables = make_valid_tables();
    tables.param_subscribers[0] = {ToSubmodel{0, 0}, ToSubmodel{9, 0}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        import_blob(tables),
        ov::Exception,
        "parameter subscriber refers to submodel 9 while only 1 submodel(s) are present");
}

// The subscriber map's key (a global input index) is just as untrusted as its values -
// bind_global_params() indexes m_npuw_model->inputs()[param_idx] with it unchecked.
TEST(NpuwImportRoutingValidation, ParamSubscriberKeyOutOfRange) {
    auto tables = make_valid_tables();
    tables.param_subscribers[9] = {ToSubmodel{0, 0}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        import_blob(tables),
        ov::Exception,
        "parameter subscriber refers to global input 9 while the model has only 1 input(s)");
}

// A dangling function-body reference is a second unchecked index into m_compiled_submodels.
TEST(NpuwImportRoutingValidation, ReplacedBySubmodelIndexOutOfRange) {
    OV_EXPECT_THROW_HAS_SUBSTRING(
        import_blob(make_valid_tables(), {SubmodelSpec{std::optional<std::size_t>{0x100000u}, {}}}),
        ov::Exception,
        "submodel 0 is replaced by submodel 1048576 while only 1 submodel(s) are present");
}

// The dangling replaced_by is also dereferenced by reconstruct_closure(), which runs
// as part of consume_weights_bank() - i.e. before validate_import_routing_tables() used
// to run. A non-empty (bank-backed) closure entry makes reconstruct_closure() actually
// resolve that index via submodel_device(), so this only passes if validation happens
// before consume_weights_bank(), not after it.
TEST(NpuwImportRoutingValidation, ReplacedBySubmodelIndexOutOfRangeWithClosureEntry) {
    OV_EXPECT_THROW_HAS_SUBSTRING(
        import_blob(make_valid_tables(), {SubmodelSpec{std::optional<std::size_t>{0x100000u}, {0}}}),
        ov::Exception,
        "submodel 0 is replaced by submodel 1048576 while only 1 submodel(s) are present");
}

// NO_LINK is a legal placeholder for a global input which no submodel consumes,
// but it is dereferenced unchecked everywhere else.
TEST(NpuwImportRoutingValidation, NoLinkIsAcceptedForGlobalInputsOnly) {
    auto tables = make_valid_tables();
    tables.inputs = {CM::NO_LINK};
    EXPECT_NO_THROW(import_blob(tables));

    tables = make_valid_tables();
    tables.outputs = {CM::NO_LINK};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "global output mapping is not linked to any submodel");

    tables = make_valid_tables();
    tables.links = {{CM::NO_LINK, ToSubmodel{0, 0}}};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "submodel input link is not linked to any submodel");

    tables = make_valid_tables();
    tables.links = {{ToSubmodel{0, 0}, CM::NO_LINK}};
    OV_EXPECT_THROW_HAS_SUBSTRING(import_blob(tables),
                                  ov::Exception,
                                  "submodel output link is not linked to any submodel");
}

// Port indices can only be validated against a submodel that carries a compiled
// blob, which an import test cannot produce without a real NPU device - so these
// two cases exercise the validation directly.
using SubmodelPorts = CM::SubmodelPorts;

std::vector<SubmodelPorts> make_submodel_ports(std::size_t count,
                                               std::size_t num_inputs = 2,
                                               std::size_t num_outputs = 1) {
    std::vector<SubmodelPorts> submodels(count);
    for (auto& ports : submodels) {
        ports.num_inputs = num_inputs;
        ports.num_outputs = num_outputs;
    }
    return submodels;
}

void validate(const std::vector<SubmodelPorts>& submodels, const Tables& t) {
    CM::validate_import_routing_tables(submodels, t.inputs.size(), t.outputs.size(), t.inputs, t.outputs,
                                       t.param_subscribers, t.links);
}

// An in-range replaced_by is not enough: request construction dereferences the
// target's compiled_model unconditionally, so the target must actually be a compiled
// function body (i.e. have known ports), not an optimized-out submodel or another
// call site missing its own compiled model.
TEST(NpuwImportRoutingValidation, ReplacedByTargetIsNotCompiledFunctionBody) {
    auto submodels = make_submodel_ports(2);
    submodels[0].num_inputs.reset();
    submodels[0].num_outputs.reset();
    submodels[1].replaced_by = 0;

    OV_EXPECT_THROW_HAS_SUBSTRING(validate(submodels, make_valid_tables()),
                                  ov::Exception,
                                  "submodel 1 is replaced by submodel 0 which is not a compiled function body");
}

TEST(NpuwImportRoutingValidation, LinkConsumerPortIndexOutOfRange) {
    auto tables = make_valid_tables();
    // Submodels have 2 inputs, so port 7 does not exist
    tables.links = {{ToSubmodel{0, 7}, ToSubmodel{0, 0}}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        validate(make_submodel_ports(1), tables),
        ov::Exception,
        "submodel input link refers to input port 7 of submodel 0 which has only 2 of them");
}

TEST(NpuwImportRoutingValidation, LinkProducerPortIndexOutOfRange) {
    auto tables = make_valid_tables();
    // Submodels have 1 output, so port 3 does not exist
    tables.links = {{ToSubmodel{0, 1}, ToSubmodel{0, 3}}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        validate(make_submodel_ports(1), tables),
        ov::Exception,
        "submodel output link refers to output port 3 of submodel 0 which has only 1 of them");
}

TEST(NpuwImportRoutingValidation, GlobalInputMappingPortIndexOutOfRange) {
    auto tables = make_valid_tables();
    // Submodels have 2 inputs, so port 7 does not exist
    tables.inputs = {ToSubmodel{0, 7}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        validate(make_submodel_ports(1), tables),
        ov::Exception,
        "global input mapping refers to input port 7 of submodel 0 which has only 2 of them");
}

TEST(NpuwImportRoutingValidation, GlobalOutputMappingPortIndexOutOfRange) {
    auto tables = make_valid_tables();
    // Submodels have 1 output, so port 3 does not exist
    tables.outputs = {ToSubmodel{0, 3}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        validate(make_submodel_ports(1), tables),
        ov::Exception,
        "global output mapping refers to output port 3 of submodel 0 which has only 1 of them");
}

// A subscriber entry is dereferenced the same way as any other link - NO_LINK is only
// legal for the global input table itself, never for what it fans out to.
TEST(NpuwImportRoutingValidation, NoLinkRejectedForParamSubscriberEntry) {
    auto tables = make_valid_tables();
    tables.param_subscribers[0] = {CM::NO_LINK};
    OV_EXPECT_THROW_HAS_SUBSTRING(validate(make_submodel_ports(1), tables),
                                  ov::Exception,
                                  "parameter subscriber is not linked to any submodel");
}

TEST(NpuwImportRoutingValidation, ParamSubscriberPortIndexOutOfRange) {
    auto tables = make_valid_tables();
    // Submodels have 2 inputs, so port 7 does not exist
    tables.param_subscribers[0] = {ToSubmodel{0, 7}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        validate(make_submodel_ports(1), tables),
        ov::Exception,
        "parameter subscriber refers to input port 7 of submodel 0 which has only 2 of them");
}

// A funcall submodel carries no ports of its own - they must be resolved
// through the function body it is replaced by.
TEST(NpuwImportRoutingValidation, FuncallPortsAreCheckedAgainstFunctionBody) {
    auto submodels = make_submodel_ports(2);
    submodels[1].num_inputs.reset();
    submodels[1].num_outputs.reset();
    submodels[1].replaced_by = 0;

    auto tables = make_valid_tables();
    tables.links = {{ToSubmodel{1, 1}, ToSubmodel{0, 0}}};
    EXPECT_NO_THROW(validate(submodels, tables));

    tables.links = {{ToSubmodel{1, 42}, ToSubmodel{0, 0}}};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        validate(submodels, tables),
        ov::Exception,
        "submodel input link refers to input port 42 of submodel 1 which has only 2 of them");
}

}  // namespace
