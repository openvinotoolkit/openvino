// Copyright (C) 2026 Fujitsu
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {

using InputShapes = std::vector<InputShape>;
using PagedAttnTestParams = std::tuple<ElementType, InputShapes, bool, bool, bool, int32_t, ov::AnyMap, bool>;

class PagedAttnTestBase : public testing::WithParamInterface<PagedAttnTestParams>, virtual public ov::test::SubgraphBaseTest,public CPUTestsBase {

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
        template <typename IT, typename T>
        static void strided_iota(IT first, size_t n, T value, T stride);
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