// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Per-architecture conversion tests for the native .gguf builder path (FrontEnd::load(path) ->
// build_ggml_graph_from_gguf -> GgufBuilderDecoder -> convert).
//
// The single-op tests in test_ops.cpp cover translators in isolation; nothing there exercises
// architecture detection, the GGUF metadata reader, or the builder's whole-graph assembly.  This
// file does, over every architecture llama.cpp can emit a model for -- 101 fixtures -- and asserts a
// definite outcome for each:
//
//   convert  the graph builds, and its fingerprint (op count / input count) matches the pinned
//            value, so a refactor that quietly restructures a supported architecture is caught.
//   reject   the architecture is not on the builder's accept list, so conversion must fail with
//            that specific diagnostic.  This turns "unsupported" from a claim in a doc into a
//            machine-checked fact, and distinguishes a clean rejection from a crash or -- worse --
//            a silent success producing a wrong graph.
//   broken   a supported architecture that currently does NOT convert.  The test asserts it still
//            fails, so the defect is recorded rather than forgotten, and FIXING it makes this test
//            fail with an instruction to promote the fixture to `convert`.  An xfail that silently
//            passes is how a known-broken list rots into a permanent excuse list.
//
// == Fixtures ==
// test_data/arch_fixtures/*.gguf.hdr are GGUF *headers* only: the bytes before the first tensor's
// data offset, i.e. magic + KV metadata + tensor table.  That is the whole input to architecture
// detection and graph construction; the weight bytes after it do not affect the converted graph's
// structure.  Each is turned back into a loadable .gguf here by appending the manifest's recorded
// number of zero bytes.  Full models would be 559 MB; the headers are ~25 KB each.
//
// The headers are GENERATED, not committed: tests/gen_arch_fixtures.py builds llama.cpp's
// test-llama-archs at a pinned commit and strips its output.  CI does this in the Linux build job
// (.github/workflows/job_build_linux.yml) and ships the result in the tests artifact.  The
// manifest, by contrast, IS committed -- it carries the reviewed expectation per architecture.
//
// So on a platform where the generator did not run, the manifest is present but its fixtures are
// not.  That is a legitimate configuration and the suite skips itself (see present_fixture_count());
// a manifest entry missing its file while OTHER fixtures exist is a broken generation and fails.
// Running the generator locally into test_data/arch_fixtures makes the suite live in-tree too.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/util/file_util.hpp"

using namespace ov_gguf_test;

namespace {

enum class Expectation { Convert, Reject, Broken };

struct ArchFixture {
    std::string header_file;  // e.g. "llama-dense.gguf.hdr"
    size_t data_bytes = 0;    // zero bytes to append to rebuild a loadable .gguf
    Expectation expectation = Expectation::Reject;
};

std::string fixture_dir() {
    return ov::util::path_join({test_data_dir(), "arch_fixtures"}).string();
}

// Parse test_data/arch_fixtures/manifest.txt: "<file> <data_bytes> <expectation>", '#' comments.
std::vector<ArchFixture> read_manifest() {
    const std::string path = ov::util::path_join({fixture_dir(), "manifest.txt"}).string();
    std::ifstream in(path);
    if (!in) {
        // Returning empty would silently reduce this file to zero tests, so make it loud.  The
        // suite below turns this into a failure via ManifestIsPresentAndComplete.
        return {};
    }
    std::vector<ArchFixture> fixtures;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ls(line);
        ArchFixture f;
        std::string expectation;
        if (!(ls >> f.header_file >> f.data_bytes >> expectation)) {
            continue;
        }
        if (expectation == "convert") {
            f.expectation = Expectation::Convert;
        } else if (expectation == "reject") {
            f.expectation = Expectation::Reject;
        } else if (expectation == "broken") {
            f.expectation = Expectation::Broken;
        } else {
            continue;  // unknown keyword; ManifestIsPresentAndComplete catches the resulting gap
        }
        fixtures.push_back(f);
    }
    return fixtures;
}

// Whether the generated *.gguf.hdr fixtures are present at all.
//
// Counted rather than checked as a boolean so a PARTIAL generation is distinguishable from no
// generation: none present means the generator did not run on this platform (skip), some present
// means it ran and produced an incomplete set (a failure, asserted in ManifestIsPresentAndComplete).
size_t count_present_fixtures(const std::vector<ArchFixture>& fixtures) {
    size_t present = 0;
    for (const auto& f : fixtures) {
        if (ov::util::file_exists(ov::util::path_join({fixture_dir(), f.header_file}).string())) {
            ++present;
        }
    }
    return present;
}

// Cached: every parameterized test consults this, and there are ~101 of them.
size_t present_fixture_count() {
    static const size_t count = count_present_fixtures(read_manifest());
    return count;
}

// Reconstruct a loadable .gguf from a header fixture: header bytes followed by data_bytes zeros.
//
// Zero-filling is exact for these fixtures, not an approximation: every tensor llama.cpp's model
// saver writes here is F32, so no dequantization is involved and no block-scale field can be made
// invalid by a zero.  The values themselves are irrelevant -- this test asserts graph structure, and
// the converted graph is built from the metadata and tensor table alone.
std::string materialize(const ArchFixture& fixture, const std::string& dir) {
    const std::string src = ov::util::path_join({fixture_dir(), fixture.header_file}).string();
    std::ifstream in(src, std::ios::binary);
    if (!in) {
        return {};
    }
    std::vector<char> header((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // Drop the ".hdr" suffix so the path ends in .gguf, which the frontend requires.
    std::string name = fixture.header_file;
    const std::string suffix = ".hdr";
    if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        name.resize(name.size() - suffix.size());
    }
    const std::string dst = ov::util::path_join({dir, name}).string();

    std::ofstream out(dst, std::ios::binary);
    if (!out) {
        return {};
    }
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    // Chunked zero-fill: a fixture's data section can be tens of MB.
    const std::vector<char> zeros(64 * 1024, 0);
    size_t remaining = fixture.data_bytes;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, zeros.size());
        out.write(zeros.data(), static_cast<std::streamsize>(chunk));
        remaining -= chunk;
    }
    out.close();
    return out ? dst : std::string{};
}

// Graph fingerprints for the fixtures expected to convert: {op count, input count}.
//
// Pinned so that a refactor which restructures a supported architecture's graph has to be an
// explicit, reviewed change to these numbers rather than an invisible one.  They are a structural
// signature, not a correctness claim -- numerical accuracy per architecture is a separate tier that
// needs real checkpoints (docs/testing_architecture.md).
struct Fingerprint {
    size_t ops;
    size_t inputs;
};

const std::map<std::string, Fingerprint>& fingerprints() {
    static const std::map<std::string, Fingerprint> fp{
        {"bailingmoe2-moe.gguf.hdr", {457, 10}}, {"ernie4_5-moe-moe.gguf.hdr", {431, 10}},
        {"exaone4-dense.gguf.hdr", {406, 11}},   {"gemma-dense.gguf.hdr", {349, 10}},
        {"gemma2-dense.gguf.hdr", {403, 11}},    {"glm4moe-moe.gguf.hdr", {469, 10}},
        {"gpt-oss-moe.gguf.hdr", {562, 11}},     {"hunyuan-dense-dense.gguf.hdr", {400, 10}},
        {"hunyuan-moe-moe.gguf.hdr", {518, 10}}, {"llama-dense.gguf.hdr", {384, 10}},
        {"llama-moe.gguf.hdr", {490, 10}},       {"maincoder-dense.gguf.hdr", {416, 10}},
        {"minicpm-dense.gguf.hdr", {382, 10}},   {"minicpm-moe.gguf.hdr", {488, 10}},
        {"minimax-m2-moe.gguf.hdr", {518, 10}},  {"mistral3-dense.gguf.hdr", {384, 10}},
        {"mistral3-moe.gguf.hdr", {490, 10}},    {"olmoe-moe.gguf.hdr", {518, 10}},
        {"phi3-dense.gguf.hdr", {364, 10}},      {"qwen2-dense.gguf.hdr", {352, 10}},
        {"qwen3-dense.gguf.hdr", {400, 10}},     {"qwen3moe-moe.gguf.hdr", {534, 10}},
        {"qwen35-dense.gguf.hdr", {391, 10}},    {"smollm3-dense.gguf.hdr", {368, 10}},
    };
    return fp;
}

// A scratch directory that outlives one test, so the reconstructed .gguf can be written once per
// test and removed afterwards without leaving multi-MB files behind on failure.
class ScratchDir {
public:
    ScratchDir() : m_path(ov::test::utils::generateTestFilePrefix() + "_gguf_arch") {
        ov::util::create_directory_recursive(std::filesystem::path(m_path));
    }
    ~ScratchDir() {
        ov::test::utils::removeDir(m_path);
    }
    const std::string& path() const {
        return m_path;
    }

private:
    std::string m_path;
};

class GGUFArchConversion : public ::testing::TestWithParam<ArchFixture> {};

TEST_P(GGUFArchConversion, MatchesManifestExpectation) {
    const ArchFixture fixture = GetParam();
    if (present_fixture_count() == 0) {
        GTEST_SKIP() << "no arch fixtures in " << fixture_dir()
                     << " -- generate them with tests/gen_arch_fixtures.py --fetch";
    }
    ScratchDir scratch;
    const std::string model_path = materialize(fixture, scratch.path());
    ASSERT_FALSE(model_path.empty()) << "could not materialize fixture " << fixture.header_file;

    ov::frontend::gguf::FrontEnd fe;
    std::shared_ptr<ov::Model> model;
    std::string error;
    try {
        model = fe.convert(fe.load(model_path));
    } catch (const std::exception& e) {
        error = e.what();
    }
    ov::test::utils::removeFile(model_path);

    switch (fixture.expectation) {
    case Expectation::Convert: {
        ASSERT_TRUE(model) << "expected " << fixture.header_file << " to convert, but it failed:\n" << error;
        const auto it = fingerprints().find(fixture.header_file);
        ASSERT_NE(it, fingerprints().end())
            << fixture.header_file << " is marked `convert` in the manifest but has no fingerprint entry in "
            << "test_arch_conversion.cpp; add one.";
        EXPECT_EQ(model->get_ops().size(), it->second.ops)
            << "graph op count for " << fixture.header_file << " changed. If the new structure is intended, "
            << "update fingerprints() -- and say why in the commit message.";
        EXPECT_EQ(model->inputs().size(), it->second.inputs)
            << "graph input count for " << fixture.header_file << " changed; see fingerprints().";
        break;
    }
    case Expectation::Reject: {
        // The accept list must be what rejects it: a different failure means the file broke the
        // reader or the builder somewhere it should never have reached.
        ASSERT_FALSE(model) << fixture.header_file << " is marked `reject` but converted successfully. If the "
                            << "builder now supports this architecture, add it to verified/experimental_archs() "
                            << "and change the manifest line to `convert` with a fingerprint.";
        EXPECT_NE(error.find("native GGUF builder does not support architecture"), std::string::npos)
            << fixture.header_file << " failed for a reason other than the architecture accept list, which means "
            << "it got further into the builder than it should. Actual error:\n"
            << error;
        break;
    }
    case Expectation::Broken: {
        // XPASS is a failure on purpose: a `broken` entry that starts converting must be promoted,
        // not left to accumulate as a permanent excuse.
        ASSERT_FALSE(model) << fixture.header_file << " is marked `broken` but now converts. The defect is fixed: "
                            << "change its manifest line to `convert` and add its fingerprint to fingerprints().";
        // Not an accept-list rejection -- that is what makes it a defect rather than an unsupported
        // architecture.
        EXPECT_EQ(error.find("native GGUF builder does not support architecture"), std::string::npos)
            << fixture.header_file << " is marked `broken` but is actually rejected by the accept list; it should "
            << "be `reject`.";
        break;
    }
    }
}

std::string fixture_test_name(const ::testing::TestParamInfo<ArchFixture>& info) {
    // "llama-dense.gguf.hdr" -> "llama_dense"; gtest names allow only [A-Za-z0-9_].
    std::string name = info.param.header_file;
    const auto dot = name.find(".gguf");
    if (dot != std::string::npos) {
        name.resize(dot);
    }
    for (auto& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
            c = '_';
        }
    }
    return name;
}

INSTANTIATE_TEST_SUITE_P(GGUFArchs, GGUFArchConversion, ::testing::ValuesIn(read_manifest()), fixture_test_name);

// The manifest is read at static-init time to instantiate the suite above, so a missing or truncated
// manifest would quietly produce zero tests instead of a failure.  Assert its shape explicitly.
TEST(GGUFArchConversionManifest, IsPresentAndComplete) {
    const auto fixtures = read_manifest();
    ASSERT_FALSE(fixtures.empty()) << "no fixtures parsed from " << fixture_dir()
                                   << "/manifest.txt -- is the test_data directory installed?";
    // Guards against a manifest that parses but lost most of its lines.
    EXPECT_GE(fixtures.size(), 90u) << "manifest has far fewer fixtures than the ~101 architectures llama.cpp "
                                    << "emits; regenerate with gen_arch_fixtures.py";

    // Fixture files are generated, so "none present" is a valid platform configuration and only
    // means the suite above skips.  "Some present" is not: it means the generator ran and produced
    // an incomplete set, which would silently shrink coverage to whatever happened to be written.
    const size_t present = present_fixture_count();
    if (present != 0) {
        for (const auto& f : fixtures) {
            const std::string path = ov::util::path_join({fixture_dir(), f.header_file}).string();
            EXPECT_TRUE(ov::util::file_exists(path))
                << "manifest lists " << f.header_file << " but the file is missing, while " << present << " of "
                << fixtures.size() << " fixtures are present -- fixture generation was incomplete. Re-run "
                << "tests/gen_arch_fixtures.py (CI: the 'Generate GGUF arch fixtures' build step).";
        }
    } else {
        GTEST_LOG_(INFO) << "no fixture files in " << fixture_dir()
                         << "; the GGUFArchs suite is skipped on this platform";
    }

    // Manifest/fingerprint consistency is checked whether or not the fixtures were generated: both
    // are committed source, so a mismatch is a bug on every platform.
    size_t convertible = 0;
    for (const auto& f : fixtures) {
        if (f.expectation == Expectation::Convert) {
            ++convertible;
            EXPECT_NE(fingerprints().count(f.header_file), 0u)
                << f.header_file << " is `convert` but has no fingerprint entry";
        }
    }
    EXPECT_GT(convertible, 0u) << "no fixture is expected to convert, which would make this suite vacuous";

    // Every fingerprint must correspond to a live manifest entry, so stale entries cannot linger
    // after a fixture is dropped upstream.
    for (const auto& entry : fingerprints()) {
        bool found = false;
        for (const auto& f : fixtures) {
            if (f.header_file == entry.first && f.expectation == Expectation::Convert) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "fingerprints() has a stale entry for " << entry.first
                           << " (no matching `convert` line in the manifest)";
    }
}

}  // namespace
