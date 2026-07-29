// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Functional tests for ov::Core's device-name dispatch: two candidate libraries registered
// under one device name (a "dispatch group"), each exporting a scriptable enumeration probe.
// Exercises the real dlopen + probe + merge + resolve path, not just the algorithm.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

#ifndef OPENVINO_STATIC_LIBRARY

namespace {

// PROBE_SCORE_* values (kept literal here so the test states expectations explicitly).
constexpr int INCOMPATIBLE = 0;
constexpr int SERVABLE = 1;
constexpr int CAPABLE = 50;
constexpr int PREFERRED = 100;

void set_env(const std::string& name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    ::setenv(name.c_str(), value.c_str(), 1);
#endif
}
void unset_env(const std::string& name) {
#ifdef _WIN32
    _putenv_s(name.c_str(), "");
#else
    ::unsetenv(name.c_str());
#endif
}

std::filesystem::path candidate_lib(const std::string& suffix) {
    return ov::util::make_plugin_library_name(ov::util::make_path(ov::test::utils::getExecutableDirectory()),
                                              "mock_dispatch_candidate_" + suffix + OV_BUILD_POSTFIX);
}

// A dispatch-group test fixture: writes a 2-<location> plugins.xml and scripts both probes.
class DispatchGroupTest : public ::testing::Test {
protected:
    const std::string device = "FAKE";
    std::filesystem::path xml_path{"test_dispatch_group_plugins.xml"};

    void write_registry() {
        std::ofstream file(xml_path);
        file << "<ie><plugins><plugin name=\"" << device << "\">"
             << "<location>" << candidate_lib("a").string() << "</location>"
             << "<location>" << candidate_lib("b").string() << "</location>"
             << "</plugin></plugins></ie>";
    }

    // Script one triple "id,fingerprint,score" per candidate.
    void script(const std::string& a_enum, const std::string& b_enum) {
        set_env("MOCK_DISPATCH_A_ENUM", a_enum);
        set_env("MOCK_DISPATCH_B_ENUM", b_enum);
    }

    void SetUp() override {
        write_registry();
    }
    void TearDown() override {
        unset_env("MOCK_DISPATCH_A_ENUM");
        unset_env("MOCK_DISPATCH_B_ENUM");
        unset_env("OV_GPU_RUNTIME");
        if (ov::util::file_exists(xml_path))
            std::ignore = std::filesystem::remove(xml_path);
    }

    // Tag ("A"/"B") of the candidate Core constructed for the bare (default) device.
    std::string resolved_tag(ov::Core& core) {
        return core.get_property(device, "MOCK_CANDIDATE_TAG").as<std::string>();
    }
    // Tag of the candidate Core constructed for a specific canonical id (e.g. "FAKE.1").
    std::string resolved_tag(ov::Core& core, const std::string& id) {
        return core.get_property(device + "." + id, "MOCK_CANDIDATE_TAG").as<std::string>();
    }
    // available_devices entries that belong to this dispatch group.
    std::vector<std::string> group_devices(ov::Core& core) {
        std::vector<std::string> group;
        for (const auto& d : core.get_available_devices())
            if (d == device || d.rfind(device + ".", 0) == 0)
                group.push_back(d);
        return group;
    }
};

// Same device (equal fingerprint) seen by both: higher score wins, only that one is built.
TEST_F(DispatchGroupTest, higher_score_wins) {
    script("0,aa," + std::to_string(PREFERRED), "0,aa," + std::to_string(CAPABLE));
    ov::Core core;
    core.register_plugins(xml_path.string());
    EXPECT_EQ(resolved_tag(core), "A");
}

TEST_F(DispatchGroupTest, higher_score_wins_other_candidate) {
    script("0,aa," + std::to_string(SERVABLE), "0,aa," + std::to_string(PREFERRED));
    ov::Core core;
    core.register_plugins(xml_path.string());
    EXPECT_EQ(resolved_tag(core), "B");
}

// Tie on score -> first registry candidate (A, listed first) wins.
TEST_F(DispatchGroupTest, tie_resolves_to_registry_order) {
    script("0,aa," + std::to_string(SERVABLE), "0,aa," + std::to_string(SERVABLE));
    ov::Core core;
    core.register_plugins(xml_path.string());
    EXPECT_EQ(resolved_tag(core), "A");
}

// INCOMPATIBLE(0) is excluded even if listed first; the other candidate serves the device.
TEST_F(DispatchGroupTest, incompatible_candidate_excluded) {
    script("0,aa," + std::to_string(INCOMPATIBLE), "0,aa," + std::to_string(SERVABLE));
    ov::Core core;
    core.register_plugins(xml_path.string());
    EXPECT_EQ(resolved_tag(core), "B");
}

// Only one candidate enumerates the device (divergent lists) -> it serves it.
TEST_F(DispatchGroupTest, single_candidate_device_served_by_that_candidate) {
    script("", "0,cc," + std::to_string(SERVABLE));
    ov::Core core;
    core.register_plugins(xml_path.string());
    EXPECT_EQ(resolved_tag(core), "B");
}

// Neither candidate serves any device -> resolution fails cleanly.
TEST_F(DispatchGroupTest, no_candidate_serves_throws) {
    script("", "");
    ov::Core core;
    core.register_plugins(xml_path.string());
    OV_EXPECT_THROW(std::ignore = resolved_tag(core), ov::Exception, ::testing::_);
}

// Two devices, opposite per-id winners: GPU.0 -> A, GPU.1 -> B, coexisting.
TEST_F(DispatchGroupTest, per_id_routing_picks_distinct_winners) {
    // device aa: A preferred, B capable; device bb: A servable, B preferred.
    script("0,aa," + std::to_string(PREFERRED) + ";1,bb," + std::to_string(SERVABLE),
           "0,aa," + std::to_string(CAPABLE) + ";1,bb," + std::to_string(PREFERRED));
    ov::Core core;
    core.register_plugins(xml_path.string());
    EXPECT_EQ(resolved_tag(core, "0"), "A");
    EXPECT_EQ(resolved_tag(core, "1"), "B");
}

// The merged canonical list has each physical device once, id-qualified.
TEST_F(DispatchGroupTest, available_devices_merged_canonical_list) {
    script("0,aa," + std::to_string(PREFERRED) + ";1,bb," + std::to_string(SERVABLE),
           "0,aa," + std::to_string(CAPABLE) + ";1,bb," + std::to_string(PREFERRED));
    ov::Core core;
    core.register_plugins(xml_path.string());
    EXPECT_EQ(group_devices(core), (std::vector<std::string>{device + ".0", device + ".1"}));
}

// Divergent enumeration: a device only one candidate sees still appears once, served by it.
TEST_F(DispatchGroupTest, divergent_enumeration_non_shared_device) {
    // Both see aa; only B sees cc (e.g. a non-Intel GPU only the OCL-like candidate lists).
    script("0,aa," + std::to_string(PREFERRED),
           "0,aa," + std::to_string(CAPABLE) + ";1,cc," + std::to_string(SERVABLE));
    ov::Core core;
    core.register_plugins(xml_path.string());
    EXPECT_EQ(group_devices(core), (std::vector<std::string>{device + ".0", device + ".1"}));
    EXPECT_EQ(resolved_tag(core, "0"), "A");  // shared device -> higher score (A)
    EXPECT_EQ(resolved_tag(core, "1"), "B");  // B-only device -> B
}

}  // namespace

#endif  // OPENVINO_STATIC_LIBRARY
