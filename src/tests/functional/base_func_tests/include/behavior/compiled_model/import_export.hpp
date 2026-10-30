// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <fstream>
#include <functional>

#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "pugixml.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace behavior {

typedef std::tuple<ov::element::Type_t,  // Element type
                   std::string,          // Device name
                   ov::AnyMap            // Config
                   >
    OVCompiledGraphImportExportTestParams;

class OVCompiledGraphImportExportTest : public testing::WithParamInterface<OVCompiledGraphImportExportTestParams>,
                                        public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<OVCompiledGraphImportExportTestParams> obj);

    void SetUp() override;

    void TearDown() override;

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    ov::element::Type elementType;
    std::shared_ptr<ov::Model> function;
};

using OVClassCompiledModelImportExportTestP = OVCompiledModelClassBaseTestP;

//
// GetRuntimeModel
//
typedef std::tuple<ov::element::Type,  // Element type
                   ov::Shape,          // Shape
                   std::string         // Device name
                   >
    OVCompiledModelGraphUniqueNodeNamesTestParams;

class OVCompiledModelGraphUniqueNodeNamesTest
    : public testing::WithParamInterface<OVCompiledModelGraphUniqueNodeNamesTestParams>,
      public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<OVCompiledModelGraphUniqueNodeNamesTestParams> obj);
    void SetUp() override;

protected:
    std::shared_ptr<ov::Model> model;
};

class OVExecGraphSerializationTest : public testing::WithParamInterface<std::string>, public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
    void TearDown() override;

private:
    // walker traverse (DFS) xml document and store layer & data nodes in
    // vector which is later used for comparison
    struct exec_graph_walker : pugi::xml_tree_walker {
        std::vector<pugi::xml_node> nodes;
        bool for_each(pugi::xml_node& node) override;
    };

    // compare_docs() helper
    std::pair<bool, std::string> compare_nodes(const pugi::xml_node& node1, const pugi::xml_node& node2);

protected:
    // checks if two exec graph xml's are equivalent:
    // - the same count of <layer> and <data> nodes
    // - the same count of attributes of each node
    // - the same name of each attribute (value is not checked, since it can differ
    // beetween different devices)
    std::pair<bool, std::string> compare_docs(const pugi::xml_document& doc1, const pugi::xml_document& doc2);

    std::string m_out_xml_path, m_out_bin_path;
};

struct HsmEntry {
    ov::runtime::hsm::DeviceId device;
    ov::runtime::hsm::SectionTagReserved tag;
    std::vector<std::byte> payload;
};

/// Rebuilds an HSM blob from its sections after @p modify edited them; throws if @p blob is not HSM.
std::string hsm_repack(const std::string& blob, const std::function<void(std::vector<HsmEntry>&)>& modify);

/// Returns a copy of @p blob with its header's format minor version set to @p minor.
std::string hsm_with_minor_version(std::string blob, uint8_t minor);

bool is_hsm(const std::string& blob);

using HsmBlobTestParams = std::tuple<std::string,  // device name
                                     ov::AnyMap    // device configuration
                                     >;

/// Same checks as OVCompiledGraphImportExportTest, for plugins whose export_model() emits HSM instead of the
/// legacy blob format (skipped otherwise) - kept alongside it since one is meant to eventually replace the other.
class OVCompiledModelHsmBlobTest : public testing::WithParamInterface<HsmBlobTestParams>,
                                   public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<HsmBlobTestParams>& obj);

    void SetUp() override;
    void TearDown() override;

protected:
    ov::CompiledModel compile(bool with_cache);
    std::string export_blob(ov::CompiledModel compiled) const;
    std::string read_cache_blob() const;
    void write_cache_blob(const std::string& blob) const;
    void expect_same_inference(ov::CompiledModel& actual);

    ov::Core core;
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> model;
    std::filesystem::path cache_path;
    ov::Tensor input;
    ov::Tensor reference;
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
