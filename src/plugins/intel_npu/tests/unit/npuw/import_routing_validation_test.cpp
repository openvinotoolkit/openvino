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
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
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
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        return {};
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
void write_submodel(std::ostream& buffer, std::optional<std::size_t> replaced_by) {
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

        std::vector<bool> is_remote;
        std::vector<int64_t> closure_uid;
        stream & is_remote & closure_uid;

        std::vector<ov::Tensor> scales, zerops;
        std::size_t closure_size = 0u;
        std::vector<std::size_t> cpu_closure_ids;
        stream & scales & zerops & closure_size & cpu_closure_ids;
    });
}

// Produces exactly what CompiledModel::export_model() would, except the routing
// tables and the submodels are whatever the caller asks for.
std::string make_blob(const Tables& tables, const std::vector<std::optional<std::size_t>>& submodels) {
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

        for (const auto& replaced_by : submodels) {
            write_submodel(buffer, replaced_by);
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

// A single submodel acting as its own function body - the shape every test below
// uses unless it needs to forge the function-body reference itself.
const std::vector<std::optional<std::size_t>> kOneSubmodel{std::optional<std::size_t>{0u}};

void import_blob(const Tables& tables, const std::vector<std::optional<std::size_t>>& submodels = kOneSubmodel) {
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
    EXPECT_THROW(import_blob(tables), ov::Exception);
}

TEST(NpuwImportRoutingValidation, LinkProducerSubmodelIndexOutOfRange) {
    auto tables = make_valid_tables();
    tables.links = {{ToSubmodel{0, 0}, ToSubmodel{0x100000, 0}}};
    EXPECT_THROW(import_blob(tables), ov::Exception);
}

TEST(NpuwImportRoutingValidation, GlobalInputMappingOutOfRange) {
    auto tables = make_valid_tables();
    tables.inputs = {ToSubmodel{9, 0}};
    EXPECT_THROW(import_blob(tables), ov::Exception);
}

TEST(NpuwImportRoutingValidation, GlobalOutputMappingOutOfRange) {
    auto tables = make_valid_tables();
    tables.outputs = {ToSubmodel{9, 0}};
    EXPECT_THROW(import_blob(tables), ov::Exception);
}

TEST(NpuwImportRoutingValidation, ParamSubscriberOutOfRange) {
    auto tables = make_valid_tables();
    tables.param_subscribers[0] = {ToSubmodel{0, 0}, ToSubmodel{9, 0}};
    EXPECT_THROW(import_blob(tables), ov::Exception);
}

// A dangling function-body reference is a second unchecked index into m_compiled_submodels.
TEST(NpuwImportRoutingValidation, ReplacedBySubmodelIndexOutOfRange) {
    EXPECT_THROW(import_blob(make_valid_tables(), {std::optional<std::size_t>{0x100000u}}), ov::Exception);
}

// NO_LINK is a legal placeholder for a global input which no submodel consumes,
// but it is dereferenced unchecked everywhere else.
TEST(NpuwImportRoutingValidation, NoLinkIsAcceptedForGlobalInputsOnly) {
    auto tables = make_valid_tables();
    tables.inputs = {CM::NO_LINK};
    EXPECT_NO_THROW(import_blob(tables));

    tables = make_valid_tables();
    tables.outputs = {CM::NO_LINK};
    EXPECT_THROW(import_blob(tables), ov::Exception);

    tables = make_valid_tables();
    tables.links = {{CM::NO_LINK, ToSubmodel{0, 0}}};
    EXPECT_THROW(import_blob(tables), ov::Exception);

    tables = make_valid_tables();
    tables.links = {{ToSubmodel{0, 0}, CM::NO_LINK}};
    EXPECT_THROW(import_blob(tables), ov::Exception);
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
    CM::validate_import_routing_tables(submodels, t.inputs, t.outputs, t.param_subscribers, t.links);
}

TEST(NpuwImportRoutingValidation, LinkConsumerPortIndexOutOfRange) {
    auto tables = make_valid_tables();
    // Submodels have 2 inputs, so port 7 does not exist
    tables.links = {{ToSubmodel{0, 7}, ToSubmodel{0, 0}}};
    EXPECT_THROW(validate(make_submodel_ports(1), tables), ov::Exception);
}

TEST(NpuwImportRoutingValidation, LinkProducerPortIndexOutOfRange) {
    auto tables = make_valid_tables();
    // Submodels have 1 output, so port 3 does not exist
    tables.links = {{ToSubmodel{0, 1}, ToSubmodel{0, 3}}};
    EXPECT_THROW(validate(make_submodel_ports(1), tables), ov::Exception);
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
    EXPECT_THROW(validate(submodels, tables), ov::Exception);
}

}  // namespace
