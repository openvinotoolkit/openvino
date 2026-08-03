// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// A minimal dispatch-group candidate plugin for testing ov::Core's device dispatch.
// Built into two independent module libraries (tags "A"/"B" via MOCK_CANDIDATE_TAG) so
// each has its own globals and can be scripted separately. Both register under one fake
// device name. The enumeration probe returns a scriptable device list read from the env
// var MOCK_DISPATCH_<TAG>_ENUM, formatted as ';'-separated "id,fingerprint_hex,score"
// triples - e.g. "0,aa,100;1,bb,1". Empty/unset means "serves nothing".

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

#ifndef MOCK_CANDIDATE_TAG
#    define MOCK_CANDIDATE_TAG "A"
#endif

namespace {

class MockCandidatePlugin : public ov::IPlugin {
public:
    MockCandidatePlugin() {
        set_device_name("FAKE");
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::filesystem::path&,
                                                      const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string& name, const ov::AnyMap&) const override {
        if (name == ov::supported_properties.name())
            return std::vector<ov::PropertyName>{};
        if (name == ov::internal::supported_properties.name())
            return std::vector<ov::PropertyName>{};
        if (name == ov::available_devices.name())
            return std::vector<std::string>{};
        // Report which candidate answered, so a test can assert who was constructed.
        if (name == "MOCK_CANDIDATE_TAG")
            return std::string(MOCK_CANDIDATE_TAG);
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

// Parse the scripted enumeration for this candidate from its env var.
void mock_enumerate(std::vector<ov::EnumeratedDevice>& out) noexcept {
    out.clear();
    const char* script = std::getenv("MOCK_DISPATCH_" MOCK_CANDIDATE_TAG "_ENUM");
    if (!script)
        return;
    try {
        std::stringstream devices(script);
        std::string triple;
        while (std::getline(devices, triple, ';')) {
            if (triple.empty())
                continue;
            std::stringstream fields(triple);
            std::string id, fingerprint_hex, score;
            std::getline(fields, id, ',');
            std::getline(fields, fingerprint_hex, ',');
            std::getline(fields, score, ',');

            ov::EnumeratedDevice e;
            e.internal_id = id;
            for (size_t i = 0; i + 1 < fingerprint_hex.size(); i += 2)
                e.fingerprint.push_back(static_cast<uint8_t>(std::stoi(fingerprint_hex.substr(i, 2), nullptr, 16)));
            e.score = static_cast<ov::DeviceCompatibilityScore>(std::stoi(score));
            out.push_back(std::move(e));
        }
    } catch (...) {
        out.clear();  // malformed script: serve nothing rather than terminate the process
    }
}

}  // namespace

static const ov::Version version = {"0.0.0", "openvino_mock_dispatch_candidate"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(MockCandidatePlugin, version)
OV_DEFINE_PLUGIN_ENUMERATE_FUNCTION(mock_enumerate)
