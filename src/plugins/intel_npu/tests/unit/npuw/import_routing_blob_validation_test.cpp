// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/opsets/opset10.hpp"

#define private public
#include "compiled_model.hpp"
#undef private
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "moe/moe_subgraph.hpp"
#include "attn/attn_subgraph.hpp"
#include "orc.hpp"
#include "orc/schema_npuw.hpp"
#include "serialization.hpp"
#include "weights_bank.hpp"

namespace {

namespace orc = ov::npuw::orc;
using CompiledModel = ov::npuw::CompiledModel;
using ToSubmodel = std::pair<std::size_t, std::size_t>;

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

std::shared_ptr<const ov::Model> make_meta_model() {
    auto parameter = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1});
    auto result = std::make_shared<ov::opset10::Result>(parameter);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter}, "RoutingBlobModel");
}

void write_optimized_out_submodel(std::ostream& buffer) {
    orc::with_leaf_section(buffer, CompiledModel::CompiledModelDesc::kOrcType, 0u, [&] {
        auto stream = ov::npuw::s11n::Stream::writer(buffer);

        std::size_t device_index = 0u;
        bool has_compiled_model = false;
        stream & device_index & has_compiled_model;

        std::optional<std::size_t> replaced_by;
        std::size_t param_base = 0u;
        bool forced_to_fcall = false;
        int64_t minus_one = -1;
        std::optional<ov::npuw::compiled::Spatial> spatial;
        stream & replaced_by & param_base & forced_to_fcall & minus_one & minus_one & minus_one & minus_one &
            minus_one & minus_one & minus_one & minus_one & spatial;

        ov::npuw::v1::subgraphs::Context empty_context;
        ov::npuw::moe::serialize_compiled_state(empty_context, stream, nullptr);
        ov::npuw::attn::serialize_compiled_state(empty_context, stream, nullptr);

        std::vector<bool> is_remote;
        std::vector<int64_t> closure_uid;
        stream & is_remote & closure_uid;

        std::vector<ov::Tensor> scales, zerops;
        std::size_t closure_size = 0u;
        std::vector<std::size_t> cpu_closure_ids;
        stream & scales & zerops & closure_size & cpu_closure_ids;
    });
}

std::string make_forged_blob() {
    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    orc::write_file_header(buffer, orc::schema_npuw::NPUW_ORC_PARTITIONED_SCHEMA);

    orc::with_section(buffer, CompiledModel::kOrcType, CompiledModel::kOrcVersion, 0u, [&] {
        orc::with_leaf_section(buffer, orc::META_SECTION_TYPE, 0u, [&] {
            auto stream = ov::npuw::s11n::Stream::writer(buffer);
            auto model = make_meta_model();
            std::string name = model->get_friendly_name();
            auto inputs = model->inputs();
            auto outputs = model->outputs();
            stream & name & inputs & outputs;

            std::vector<ToSubmodel> input_links{{0u, 0u}};
            std::vector<ToSubmodel> output_links{{0u, 0u}};
            std::map<std::size_t, std::vector<ToSubmodel>> param_subscribers;
            std::map<ToSubmodel, ToSubmodel> links{{{0x100000u, 0u}, {0u, 0u}}};
            stream & input_links & output_links & param_subscribers & links;

            std::vector<std::string> devices{"NPU"};
            stream & devices;
            auto options = std::make_shared<::intel_npu::OptionsDesc>();
            ::intel_npu::registerNPUWOptions(*options);
            ::intel_npu::Config config(options);
            config.update({{std::string(::intel_npu::NPUW_DEVICES::key()), "NPU"}});
            stream & config;

            ov::AnyMap properties;
            bool is_weightless = false;
            ov::npuw::s11n::BF16Cache bf16_consts;
            stream & properties & is_weightless & bf16_consts;
        });

        write_optimized_out_submodel(buffer);
        orc::with_leaf_section(buffer, ov::npuw::weights::Bank::kOrcType, ov::npuw::weights::Bank::kOrcVersion, [&] {
            auto stream = ov::npuw::s11n::Stream::writer(buffer);
            std::string bank_name = "routing_blob_test";
            std::size_t bank_size = 0u;
            stream & bank_name & bank_size;
        });
    });
    return buffer.str();
}

TEST(NpuwImportRoutingBlobValidation, RejectsOutOfBoundsInterSubmodelIndexDuringImport) {
    auto plugin = std::make_shared<NullPlugin>();
    auto bytes = make_forged_blob();
    std::stringstream stream(bytes, std::ios::in | std::ios::out | std::ios::binary);

    try {
        CompiledModel::import_model(stream, plugin, {});
        FAIL() << "Expected forged routing table to be rejected during import";
    } catch (const ov::Exception& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("m_submodels_input_to_prev_output[0] input submodel index 1048576"),
                  std::string::npos)
            << message;
    }
}

}  // namespace