// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace test {

using InputShapes = std::vector<InputShape>;
using PagedAttnTestParams = std::tuple<ElementType, InputShapes, bool, bool, bool, int32_t, ov::AnyMap, bool>;

class PagedAttnTestBase : public testing::WithParamInterface<PagedAttnTestParams>, virtual public ov::test::SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnTestParams>& obj);
    static std::shared_ptr<ov::op::v0::Parameter> make_param(const PartialShape& pshape, element::Type element_type, const std::string& name);
    std::shared_ptr<ov::Model> get_model(ov::element::Type data_type, bool enable_xattn, ov::Dimension::value_type head_size = 64, ov::Dimension::value_type head_num = 8, bool use_sink_input = true, int32_t sliding_window = 0, bool add_shared_reader = false);
    virtual std::shared_ptr<ov::Model> get_ref_model(ov::element::Type data_type,
                                                    ov::Dimension::value_type head_size = 64,
                                                    ov::Dimension::value_type head_num = 8,
                                                    bool use_sink_input = true,
                                                    bool add_shared_reader = false);
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    virtual void generate(int idx,
                        const bool isPagedAttn,
                        const std::vector<ov::Shape>& targetInputStaticShapes,
                        bool extendBlockIndices,
                        bool use_sink_input = true);
    void prepare();
    void update();
    void reset();
    void init_kv_cache(size_t block_nums);
    std::vector<ov::Tensor> run_pa_inference(bool extendBlockIndices, bool sinkInput);

    std::vector<size_t> transposeOrder;
    size_t keyGroupSize = 0;
    bool quantKeyByChannel = false;
    bool hasShapeOf;
    ov::Tensor key_cache;
    ov::Tensor value_cache;
    int32_t past_len_count = 0;
    int32_t sliding_window = 0;
};

class PagedAttnVSSDPATest : public PagedAttnTestBase {
public:
    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model, bool extendBlockIndices, bool sinkInput = true);

    std::vector<ov::Tensor> run_ref_test(std::shared_ptr<ov::Model> model, bool sinkInput);
};

class PagedAttnVSMatmulTest : public PagedAttnTestBase {
public:
    std::shared_ptr<ov::Model> get_ref_model(ov::element::Type data_type,
                                             ov::Dimension::value_type head_size = 64,
                                             ov::Dimension::value_type head_num = 8,
                                             bool use_sink_input = false,
                                             bool add_shared_reader = false) override ;

    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model,
                                     bool extendBlockIndices,
                                     bool sinkInput = false);

    std::vector<ov::Tensor> run_ref_test(std::shared_ptr<ov::Model> model);
};

// Regression test: executor cache collision with mixed head_size.
// Two PA nodes with head_size=256 and head_size=512 in ONE compiled model.
// Without the fix, PA2 reuses BRGEMM kernels configured for PA1's head_size,
// producing garbage. Requires f16 KV cache to trigger the BRGEMM code path.
class PagedAttnCacheCollisionTest : public PagedAttnTestBase {
public:
    static constexpr int64_t hs2_val = 512;

    void SetUp() override;

    std::shared_ptr<ov::Model> get_mixed_head_model(ov::element::Type data_type, int64_t hs1, int64_t hn);

    void generate(int idx,
                  const bool isPagedAttn,
                  const std::vector<ov::Shape>& targetInputStaticShapes,
                  bool extendBlockIndices,
                  bool use_sink_input = true) override;

    void init_all_kv_caches(size_t block_nums);

    ov::Tensor key_cache_1;
    ov::Tensor value_cache_1;
    std::vector<std::vector<ov::Shape>> targetStaticShapes2_;
};

}
}