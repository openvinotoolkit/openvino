// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "model_builder.hpp"
#include "openvino/openvino.hpp"
#include "orc.hpp"
#include "serialization.hpp"

using ov::test::npuw::ModelBuilder;

namespace {

using Link = std::pair<std::size_t, std::size_t>;

constexpr Link NO_LINK{static_cast<std::size_t>(-1), static_cast<std::size_t>(-1)};

// Serialized layout of a std::vector<Link>: an element count followed by the
// elements themselves, all of them plain std::size_t's written as raw bytes.
constexpr std::size_t kWord = sizeof(std::size_t);

std::size_t link_at(std::size_t vector_at, std::size_t idx) {
    return vector_at + kWord + idx * 2 * kWord;
}

struct MetaLinks {
    std::size_t inputs_at = 0u;   // where m_inputs_to_submodels_inputs starts in the blob
    std::size_t outputs_at = 0u;  // where m_outputs_to_submodels_outputs starts in the blob
    std::vector<Link> inputs;
    std::vector<Link> outputs;
};

// Walks the ORC META section exactly like CompiledModel::deserialize_orc_container
// does, recording where the parameter/result link vectors live in the blob.
MetaLinks read_meta_links(const std::string& blob) {
    std::istringstream in(blob, std::ios::in | std::ios::binary);
    ov::npuw::orc::read_file_header(in);
    ov::npuw::orc::ScopedReadSection root(in);
    ov::npuw::orc::ScopedReadSection meta(in);

    auto meta_stream = ov::npuw::s11n::Stream::reader(in);
    std::string model_name;
    ov::ParameterVector parameters;
    ov::NodeVector results;
    meta_stream & model_name & parameters & results;

    MetaLinks links;
    links.inputs_at = static_cast<std::size_t>(in.tellg());
    meta_stream & links.inputs;
    links.outputs_at = static_cast<std::size_t>(in.tellg());
    meta_stream & links.outputs;
    return links;
}

void poke(std::string& blob, std::size_t at, std::size_t value) {
    ASSERT_LE(at + sizeof(value), blob.size());
    std::memcpy(&blob[at], &value, sizeof(value));
}

}  // namespace

// The parameter/result/interconnect index maps are restored verbatim from the blob,
// and at inference time their entries are used as raw subscripts into the model's and
// the subrequests' port vectors. A blob carrying out-of-range indices must be rejected
// on import instead of reading past those vectors (CWE-125).
class ImportBlobIoLinksTestNPUW : public ::testing::Test {
public:
    void SetUp() override {
        ModelBuilder mb;
        m_ov_model = mb.get_model_with_repeated_blocks_with_weightless_cache();
        m_props = {{"NPU_USE_NPUW", "YES"}, {"NPUW_DEVICES", "CPU"}, {"CACHE_MODE", "OPTIMIZE_SPEED"}};
    }

protected:
    std::string export_blob() {
        auto compiled = m_core.compile_model(m_ov_model, "NPU", m_props);
        std::stringstream blob;
        compiled.export_model(blob);
        return blob.str();
    }

    void expect_import_rejected(const std::string& blob) {
        std::stringstream stream(blob, std::ios::in | std::ios::out | std::ios::binary);
        try {
            // Without the index-map validation the malformed link is only consumed at
            // the first inference - so run one here to cover that path as well
            auto imported = m_core.import_model(stream, "NPU", m_props);
            auto request = imported.create_infer_request();
            request.infer();
            FAIL() << "Expected an out-of-range index map to be rejected";
        } catch (const ov::Exception& ex) {
            EXPECT_NE(std::string(ex.what()).find("NPU NPUW"), std::string::npos) << ex.what();
        }
    }

    std::shared_ptr<ov::Model> m_ov_model;
    ov::AnyMap m_props;
    ov::Core m_core;
};

TEST_F(ImportBlobIoLinksTestNPUW, RejectsOutOfRangeSubmodelInputIndex) {
    auto blob = export_blob();
    const auto links = read_meta_links(blob);
    ASSERT_FALSE(links.inputs.empty());

    std::size_t param_idx = links.inputs.size();
    for (std::size_t i = 0; i < links.inputs.size(); i++) {
        if (links.inputs[i] != NO_LINK) {
            param_idx = i;
            break;
        }
    }
    ASSERT_LT(param_idx, links.inputs.size()) << "No model input is linked to a subgraph input";

    // Make the parameter point at a port its consuming subgraph doesn't have
    poke(blob, link_at(links.inputs_at, param_idx) + kWord, 1024u);
    expect_import_rejected(blob);
}

TEST_F(ImportBlobIoLinksTestNPUW, RejectsOutOfRangeSubmodelIndex) {
    auto blob = export_blob();
    const auto links = read_meta_links(blob);
    ASSERT_FALSE(links.outputs.empty());

    // Make the result point at a subgraph the model doesn't have
    poke(blob, link_at(links.outputs_at, 0u), 1024u);
    expect_import_rejected(blob);
}
