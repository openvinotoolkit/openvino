// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// --- Security regression tests: CWE-787 OOB write via attacker-controlled closure indices ---
//
// Tickets:
//   CVS-193533 - non-weightless deserialize branch: `stream & closure.closure[cidx]`
//   CVS-193535 - weightless deserialize branch: `closure.closure[idx] = std::move(cpu_closures[tidx++])`
//                and `lazy_closure[idx] = std::move(non_cpu_tensors[ltidx++])`
//
// Both branches of CompiledModelDesc::serialize() read the closure size, the per-slot
// indices and the parallel is_remote/closure_uid vectors as *independent* records from
// the blob, then used the indices in std::vector::operator[] with no bounds check.
// Because the element type is ov::Tensor, the indexed store is an assignment: it
// dereferences and decrements the refcount of whatever lives at the out-of-bounds
// address before writing a new pointer there - a write-what-where primitive.
//
// The blob writer can never emit such a stream (it derives the ids from the closure it
// is writing), so these tests hand-craft the record sequence that the deserializer
// expects and drive the real production code path:
//   CompiledModelDesc::serialize(Stream::reader(...), ctx)
//
// Which branch runs is picked by ctx.is_weightless, which on import is itself derived
// from a flag stored in the blob - so both branches are attacker-reachable.

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "compiled_model.hpp"
#include "lazy_tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "serialization.hpp"

// Grants access to CompiledModel::CompiledModelDesc, which is private.
// Declared as a friend in compiled_model.hpp; every member here is test-only.
struct NpuwClosureS11nTestAccess {
    using Desc = ov::npuw::CompiledModel::CompiledModelDesc;
    using Stream = ov::npuw::s11n::Stream;

    // How a blob describes one closure. Every field is written verbatim, which is what
    // lets a test express combinations the production writer would never produce.
    struct BlobSpec {
        // Length of the is_remote / closure_uid vectors. The writer keeps these equal to
        // closure_size; a blob does not have to.
        std::size_t meta_size = 0;
        // The closure_size record - what the deserializer resizes the closure to.
        std::size_t closure_size = 0;
        // Slot indices for host-side (CPU) tensors, used as subscripts by the reader.
        std::vector<std::size_t> cpu_ids;
        // How many tensors actually follow the id vector. Equals cpu_ids.size() in a
        // well-formed blob; a short payload is its own out-of-bounds read.
        std::size_t cpu_tensor_count = 0;
        // Weightless branch only: the lazy_closure counterparts.
        std::vector<std::size_t> non_cpu_ids;
        std::size_t non_cpu_tensor_count = 0;
        // Marks slots as bank-owned (uid != -1) rather than host-side (uid == -1).
        std::vector<std::size_t> bank_owned_slots;
    };

    // What a test needs to observe after a successful round trip, extracted here so no
    // test has to name the private Desc type itself.
    struct Deserialized {
        std::size_t closure_size = 0;
        std::vector<bool> slot_filled;
        std::vector<float> slot_value;  // first element of each filled tensor, 0 otherwise
    };

    // Mirrors the field list at the top of CompiledModelDesc::serialize(). replaced_by is
    // set and compiled_model left empty so the desc reads as a function call, which skips
    // the moe/attn compiled-state records - they are irrelevant to closure handling.
    //
    // If a field is ever added to that list, the ValidBlob* tests below start failing,
    // which is the intended signal to update this helper.
    static void write_prefix(Stream& stream) {
        Desc prefix;
        prefix.replaced_by = 0u;
        stream & prefix.replaced_by & prefix.param_base & prefix.forced_to_fcall & prefix.host_gather.dst_idx &
            prefix.host_gather.src_idx & prefix.host_gather.idx_idx & prefix.quant_unpack_gather.dst_idx &
            prefix.quant_unpack_gather.src_w_idx & prefix.quant_unpack_gather.src_z_idx &
            prefix.quant_unpack_gather.src_s_idx & prefix.quant_unpack_gather.idx_idx & prefix.spatial;
    }

    static ov::Tensor make_tensor(float seed) {
        ov::Tensor tensor(ov::element::f32, ov::Shape{2});
        tensor.data<float>()[0] = seed;
        tensor.data<float>()[1] = seed + 1.0f;
        return tensor;
    }

    static void write_closure_meta(Stream& stream, const BlobSpec& spec) {
        std::vector<bool> is_remote(spec.meta_size, false);
        std::vector<int64_t> closure_uid(spec.meta_size, -1);
        for (const auto slot : spec.bank_owned_slots) {
            if (slot < closure_uid.size()) {
                closure_uid[slot] = static_cast<int64_t>(slot);
            }
        }
        stream & is_remote & closure_uid;
    }

    static std::string build_non_weightless_blob(const BlobSpec& spec) {
        std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
        auto stream = Stream::writer(buffer);
        write_prefix(stream);
        write_closure_meta(stream, spec);

        std::vector<ov::Tensor> scales;
        std::vector<ov::Tensor> zerops;
        stream & scales & zerops;

        auto closure_size = spec.closure_size;
        auto cpu_ids = spec.cpu_ids;
        stream & closure_size & cpu_ids;
        // This branch deserializes tensors straight into the slots named by cpu_ids, so
        // the payload is a bare sequence with no count record of its own.
        for (std::size_t i = 0; i < spec.cpu_tensor_count; ++i) {
            auto tensor = make_tensor(static_cast<float>(i));
            stream & tensor;
        }
        return buffer.str();
    }

    static std::string build_weightless_blob(const BlobSpec& spec) {
        std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
        auto stream = Stream::writer(buffer);
        write_prefix(stream);
        write_closure_meta(stream, spec);

        // An empty WeightsContext has no const_to_offset entries, so serialize_weightless
        // degrades to plain tensor serialization - no weights file is needed here.
        const ov::npuw::s11n::WeightsContext ctx;
        std::vector<ov::Tensor> scales;
        std::vector<ov::Tensor> zerops;
        ov::npuw::orc::serialize_weightless(stream, scales, ctx);
        ov::npuw::orc::serialize_weightless(stream, zerops, ctx);

        auto closure_size = spec.closure_size;
        auto cpu_ids = spec.cpu_ids;
        stream & closure_size & cpu_ids;

        std::vector<ov::Tensor> cpu_closures;
        for (std::size_t i = 0; i < spec.cpu_tensor_count; ++i) {
            cpu_closures.push_back(make_tensor(static_cast<float>(i)));
        }
        ov::npuw::orc::serialize_weightless(stream, cpu_closures, ctx);

        auto non_cpu_ids = spec.non_cpu_ids;
        // Empty LazyTensors serialize as a single `false` each and stay falsy on read, so
        // the deserializer will not try to resolve a weight for them.
        std::vector<ov::npuw::weights::LazyTensor> non_cpu_tensors(spec.non_cpu_tensor_count);
        stream & non_cpu_ids & non_cpu_tensors;
        return buffer.str();
    }

    // Runs the production deserializer over the crafted bytes. Throws on rejection.
    //
    // CompiledModelDesc::serialize() is bidirectional - it is the read path too. The
    // direction comes from the stream, so Stream::reader() below is what makes this a
    // deserialization; there is no separate deserialize() member to call.
    static Deserialized deserialize(const std::string& bytes, bool weightless) {
        std::stringstream buffer(bytes, std::ios::in | std::ios::out | std::ios::binary);
        auto stream = Stream::reader(buffer);
        ov::npuw::s11n::WeightsContext ctx;
        ctx.is_weightless = weightless;

        Desc desc;
        desc.serialize(stream, ctx);

        const auto& closure = desc.closure.get().closure;
        Deserialized out;
        out.closure_size = closure.size();
        for (const auto& tensor : closure) {
            out.slot_filled.push_back(static_cast<bool>(tensor));
            out.slot_value.push_back(tensor ? tensor.data<float>()[0] : 0.0f);
        }
        return out;
    }
};

namespace {

using Access = NpuwClosureS11nTestAccess;
using BlobSpec = Access::BlobSpec;

// Far enough past any real closure to land outside the allocation.
constexpr std::size_t kOutOfBoundsIndex = 0xFFFFFFFFFFFFFF00ull;

// Pins down which slots the deserializer touched: exactly those named by `expected_ids`,
// and no others. A mis-indexed store shows up here as a neighbouring slot that came back
// filled, or as a named slot that did not - neither of which the per-slot value checks
// alone would catch.
void expect_only_these_slots_filled(const Access::Deserialized& result,
                                    const std::vector<std::size_t>& expected_ids) {
    std::vector<bool> expected(result.closure_size, false);
    for (const auto slot : expected_ids) {
        ASSERT_LT(slot, result.closure_size);
        expected[slot] = true;
    }
    for (std::size_t slot = 0; slot < result.closure_size; ++slot) {
        EXPECT_EQ(result.slot_filled[slot], expected[slot]) << "slot " << slot;
    }
}

BlobSpec well_formed_non_weightless() {
    BlobSpec spec;
    spec.meta_size = 3;
    spec.closure_size = 3;
    spec.cpu_ids = {0, 1, 2};
    spec.cpu_tensor_count = 3;
    return spec;
}

BlobSpec well_formed_weightless() {
    BlobSpec spec;
    spec.meta_size = 3;
    spec.closure_size = 3;
    spec.cpu_ids = {0, 1};
    spec.cpu_tensor_count = 2;
    spec.non_cpu_ids = {2};
    spec.non_cpu_tensor_count = 1;
    spec.bank_owned_slots = {2};
    return spec;
}

}  // namespace

TEST(NpuwClosureDeserializeTest, ValidBlobNonWeightlessRoundTrips) {
    const auto spec = well_formed_non_weightless();
    Access::Deserialized result;
    OV_ASSERT_NO_THROW(result = Access::deserialize(Access::build_non_weightless_blob(spec), /*weightless=*/false));

    // closure_size is its own record: it decides the vector length independently of how
    // many tensors the payload carries and of which slots the ids name.
    ASSERT_EQ(result.closure_size, spec.closure_size);
    for (std::size_t k = 0; k < spec.cpu_ids.size(); ++k) {
        const auto slot = spec.cpu_ids[k];
        ASSERT_TRUE(result.slot_filled[slot]) << "slot " << slot;
        // Payload position k lands in slot cpu_ids[k]; make_tensor(k) seeded it with k.
        EXPECT_FLOAT_EQ(result.slot_value[slot], static_cast<float>(k));
    }
    expect_only_these_slots_filled(result, spec.cpu_ids);
}

TEST(NpuwClosureDeserializeTest, ValidBlobWeightlessRoundTrips) {
    const auto spec = well_formed_weightless();
    Access::Deserialized result;
    OV_ASSERT_NO_THROW(result = Access::deserialize(Access::build_weightless_blob(spec), /*weightless=*/true));

    ASSERT_EQ(result.closure_size, spec.closure_size);
    for (std::size_t k = 0; k < spec.cpu_ids.size(); ++k) {
        const auto slot = spec.cpu_ids[k];
        ASSERT_TRUE(result.slot_filled[slot]) << "slot " << slot;
        EXPECT_FLOAT_EQ(result.slot_value[slot], static_cast<float>(k));
    }
    // Bank-owned slots stay empty in `closure` - they are carried by lazy_closure.
    expect_only_these_slots_filled(result, spec.cpu_ids);
}

// The fixtures above use the identity ordering cpu_ids == {0, 1, ...}, under which a
// deserializer that ignored cpu_ids entirely and filled slots front to back would still
// pass. Since using those ids as subscripts is the very mechanism CVS-193533/193535 are
// about, pin the mapping down with a permutation: payload position k must land in slot
// cpu_ids[k], not in slot k.

TEST(NpuwClosureDeserializeTest, NonWeightlessClosureIdsControlSlotPlacement) {
    auto spec = well_formed_non_weightless();
    spec.cpu_ids = {2, 0, 1};

    Access::Deserialized result;
    OV_ASSERT_NO_THROW(result = Access::deserialize(Access::build_non_weightless_blob(spec), /*weightless=*/false));

    ASSERT_EQ(result.closure_size, spec.closure_size);
    EXPECT_FLOAT_EQ(result.slot_value[2], 0.0f);
    EXPECT_FLOAT_EQ(result.slot_value[0], 1.0f);
    EXPECT_FLOAT_EQ(result.slot_value[1], 2.0f);
    expect_only_these_slots_filled(result, spec.cpu_ids);
}

TEST(NpuwClosureDeserializeTest, WeightlessClosureIdsControlSlotPlacement) {
    auto spec = well_formed_weightless();
    spec.cpu_ids = {2, 0};  // payload #0 -> slot 2, payload #1 -> slot 0
    spec.non_cpu_ids = {1};
    spec.bank_owned_slots = {1};

    Access::Deserialized result;
    OV_ASSERT_NO_THROW(result = Access::deserialize(Access::build_weightless_blob(spec), /*weightless=*/true));

    ASSERT_EQ(result.closure_size, spec.closure_size);
    EXPECT_FLOAT_EQ(result.slot_value[2], 0.0f);
    EXPECT_FLOAT_EQ(result.slot_value[0], 1.0f);
    expect_only_these_slots_filled(result, spec.cpu_ids);  // slot 1 is bank-owned, stays empty
}

// --- CVS-193533: non-weightless branch ------------------------------------------------

TEST(NpuwClosureDeserializeTest, NonWeightlessOutOfBoundsClosureIdIsRejected) {
    auto spec = well_formed_non_weightless();
    spec.cpu_ids = {0, 1, kOutOfBoundsIndex};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_non_weightless_blob(spec), /*weightless=*/false),
        ov::Exception,
        "out of bounds");
}

TEST(NpuwClosureDeserializeTest, NonWeightlessClosureIdEqualToSizeIsRejected) {
    // The off-by-one case: idx == closure_size is already one element past the end.
    auto spec = well_formed_non_weightless();
    spec.cpu_ids = {0, 1, 3};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_non_weightless_blob(spec), /*weightless=*/false),
        ov::Exception,
        "out of bounds");
}

TEST(NpuwClosureDeserializeTest, NonWeightlessShortClosureMetadataIsRejected) {
    // closure_size claims 3 slots but is_remote/closure_uid only describe 2. Without the
    // check this passes deserialization and reads out of bounds later, in
    // finalize_weights_bank(), which walks closure_uid[cidx] for every cidx < closure.size().
    auto spec = well_formed_non_weightless();
    spec.meta_size = 2;
    spec.cpu_ids = {0, 1};
    spec.cpu_tensor_count = 2;
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_non_weightless_blob(spec), /*weightless=*/false),
        ov::Exception,
        "inconsistent serialized closure metadata");
}

TEST(NpuwClosureDeserializeTest, NonWeightlessHugeClosureSizeIsRejectedBeforeAllocating) {
    // The metadata check runs before closure.resize(closure_size), so an absurd size is
    // refused instead of being handed to the allocator.
    auto spec = well_formed_non_weightless();
    spec.closure_size = 1ull << 60;
    spec.cpu_ids = {};
    spec.cpu_tensor_count = 0;
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_non_weightless_blob(spec), /*weightless=*/false),
        ov::Exception,
        "inconsistent serialized closure metadata");
}

// --- CVS-193535: weightless branch ----------------------------------------------------

TEST(NpuwClosureDeserializeTest, WeightlessOutOfBoundsCpuClosureIdIsRejected) {
    auto spec = well_formed_weightless();
    spec.cpu_ids = {0, kOutOfBoundsIndex};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_weightless_blob(spec), /*weightless=*/true),
        ov::Exception,
        "out of bounds");
}

TEST(NpuwClosureDeserializeTest, WeightlessOutOfBoundsNonCpuClosureIdIsRejected) {
    // Indexes lazy_closure rather than closure - a separate unchecked subscript.
    auto spec = well_formed_weightless();
    spec.non_cpu_ids = {kOutOfBoundsIndex};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_weightless_blob(spec), /*weightless=*/true),
        ov::Exception,
        "out of bounds");
}

TEST(NpuwClosureDeserializeTest, WeightlessShortCpuPayloadIsRejected) {
    // Three ids but only two tensors follow: the fill loop would walk cpu_closures past
    // its end and move the result into a legitimate slot.
    auto spec = well_formed_weightless();
    spec.cpu_ids = {0, 1, 2};
    spec.cpu_tensor_count = 2;
    spec.non_cpu_ids = {};
    spec.non_cpu_tensor_count = 0;
    spec.bank_owned_slots = {};
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_weightless_blob(spec), /*weightless=*/true),
        ov::Exception,
        "does not match the number of serialized tensors");
}

TEST(NpuwClosureDeserializeTest, WeightlessShortNonCpuPayloadIsRejected) {
    auto spec = well_formed_weightless();
    spec.non_cpu_ids = {1, 2};
    spec.non_cpu_tensor_count = 1;
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_weightless_blob(spec), /*weightless=*/true),
        ov::Exception,
        "does not match the number of serialized tensors");
}

TEST(NpuwClosureDeserializeTest, WeightlessShortClosureMetadataIsRejected) {
    auto spec = well_formed_weightless();
    spec.meta_size = 2;
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_weightless_blob(spec), /*weightless=*/true),
        ov::Exception,
        "inconsistent serialized closure metadata");
}

TEST(NpuwClosureDeserializeTest, WeightlessHugeClosureSizeIsRejectedBeforeAllocating) {
    // Guards both closure.resize() and lazy_closure.resize() on this branch.
    auto spec = well_formed_weightless();
    spec.closure_size = 1ull << 60;
    spec.cpu_ids = {};
    spec.cpu_tensor_count = 0;
    spec.non_cpu_ids = {};
    spec.non_cpu_tensor_count = 0;
    OV_EXPECT_THROW_HAS_SUBSTRING(
        Access::deserialize(Access::build_weightless_blob(spec), /*weightless=*/true),
        ov::Exception,
        "inconsistent serialized closure metadata");
}
