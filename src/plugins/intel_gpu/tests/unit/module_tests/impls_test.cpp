// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/implementation_manager.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "test_utils.h"
#include "impls/registry/registry.hpp"
#include "primitive_type_base.h"
#include <memory>

using namespace cldnn;
using namespace ::tests;


namespace cldnn {

struct some_primitive : public primitive_base<some_primitive> {
    CLDNN_DECLARE_PRIMITIVE(some_primitive)

    enum class SomeParameter {
        SUPPORTED_VALUE_ALL,
        SUPPORTED_VALUE_ONEDNN_1,
        SUPPORTED_VALUE_ONEDNN_2,
        SUPPORTED_VALUE_OCL_STATIC,
        SUPPORTED_VALUE_OCL_DYNAMIC_1,
        SUPPORTED_VALUE_OCL_DYNAMIC,
        UNSUPPORTED_VALUE_ALL
    };

    some_primitive() : primitive_base("", {}) {}
    some_primitive(const primitive_id& id, const std::vector<input_info>& inputs, SomeParameter p) : primitive_base(id, inputs), param(p) {}

    SomeParameter param;
};

template <>
struct typed_program_node<some_primitive> : public typed_program_node_base<some_primitive> {
    using parent = typed_program_node_base<some_primitive>;
    using parent::parent;
    typed_program_node(const std::shared_ptr<some_primitive> prim, program& prog) : parent(prim, prog) { support_padding_all(true); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using some_primitive_node = typed_program_node<some_primitive>;

template <>
class typed_primitive_inst<some_primitive> : public typed_primitive_inst_base<some_primitive> {
public:

    using parent = typed_primitive_inst_base<some_primitive>;
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(some_primitive_node const& /*node*/, const kernel_impl_params& impl_param) {
        return {layout({1}, ov::element::f32, format::bfyx)};
    }
    static layout calc_output_layout(some_primitive_node const& node, kernel_impl_params const& impl_param) {
        return layout({1}, ov::element::f32, format::bfyx);
    }
    static std::string to_string(some_primitive_node const& node) { OPENVINO_NOT_IMPLEMENTED; }

public:
    using parent::parent;
};
using some_primitive_inst = typed_primitive_inst<some_primitive>;

GPU_DEFINE_PRIMITIVE_TYPE_ID(some_primitive)


struct some_impl : public typed_primitive_impl<some_primitive>  {
    using parent = typed_primitive_impl<some_primitive>;
    using parent::parent;
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::some_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<some_impl>(*this);
    }

    some_impl() : parent("some_impl") {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, some_primitive_inst& instance) override {
        return nullptr;
    }

    void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) override {}

    static std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) {
        return cldnn::make_unique<some_impl>();
    }
};

struct SomeImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("SomeImpl")
    SomeImplementationManager(shape_types shape_type, ValidateFunc vf) : ImplementationManager(impl_types::onednn, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override {
        return some_impl::create(node, params);
    }

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<some_primitive>());
        auto p = node.as<some_primitive>().get_primitive()->param;

        if (!one_of(p, some_primitive::SomeParameter::SUPPORTED_VALUE_ALL,
                       some_primitive::SomeParameter::SUPPORTED_VALUE_ONEDNN_1,
                       some_primitive::SomeParameter::SUPPORTED_VALUE_ONEDNN_2))
            return false;
        return ImplementationManager::validate(node);
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    bool support_shapes(const kernel_impl_params& params) const override {
        return get_shape_type(params) == shape_types::static_shape;
    }
};


}  // namespace cldnn

namespace ov {
namespace intel_gpu {

using namespace cldnn;

template<>
const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<some_primitive>::get_implementations() {
    static bool initialize = true;

    if (initialize) {
        implementation_map<some_primitive>::add(impl_types::ocl, shape_types::static_shape, some_impl::create, {});
        implementation_map<some_primitive>::add(impl_types::ocl, shape_types::dynamic_shape, some_impl::create, {});
        initialize = false;
    }

    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(SomeImplementationManager, shape_types::static_shape,
            [](const program_node& node) {
                auto p = node.as<some_primitive>().get_primitive()->param;
                if (one_of(p, some_primitive::SomeParameter::SUPPORTED_VALUE_ONEDNN_1))
                    return true;
                return false;
        }),
        OV_GPU_GET_INSTANCE_OCL(some_primitive, shape_types::static_shape,
            [](const program_node& node) {
                auto p = node.as<some_primitive>().get_primitive()->param;
                if (!one_of(p, some_primitive::SomeParameter::SUPPORTED_VALUE_ALL, some_primitive::SomeParameter::SUPPORTED_VALUE_OCL_STATIC))
                    return false;
                return true;
        }),
        OV_GPU_CREATE_INSTANCE_ONEDNN(SomeImplementationManager, shape_types::static_shape,
            [](const program_node& node) {
                auto p = node.as<some_primitive>().get_primitive()->param;
                if (one_of(p, some_primitive::SomeParameter::SUPPORTED_VALUE_ONEDNN_2))
                    return true;
                return false;
        }),
        OV_GPU_GET_INSTANCE_OCL(some_primitive, shape_types::dynamic_shape,
            [](const program_node& node) {
                auto p = node.as<some_primitive>().get_primitive()->param;
                if (!one_of(p, some_primitive::SomeParameter::SUPPORTED_VALUE_ALL, some_primitive::SomeParameter::SUPPORTED_VALUE_OCL_DYNAMIC))
                    return false;
                return true;
        }),
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov


TEST(impls_test, has_2_not_null_impls) {
    auto list = some_primitive::type_id()->get_all_implementations();
    ASSERT_EQ(list.size(), 4);
    for (size_t i = 0; i < list.size(); i++) {
        ASSERT_NE(list[i], nullptr) << " i = " << i;
    }

    ASSERT_EQ(list[0]->get_impl_type(), impl_types::onednn);
    ASSERT_EQ(list[1]->get_impl_type(), impl_types::ocl);
    ASSERT_EQ(list[2]->get_impl_type(), impl_types::onednn);
    ASSERT_EQ(list[3]->get_impl_type(), impl_types::ocl);

    ASSERT_EQ(list[0]->get_shape_type(), shape_types::static_shape);
    ASSERT_EQ(list[1]->get_shape_type(), shape_types::static_shape);
    ASSERT_EQ(list[2]->get_shape_type(), shape_types::static_shape);
    ASSERT_EQ(list[3]->get_shape_type(), shape_types::dynamic_shape);
}

TEST(impls_test, same_result_on_each_call) {
    auto list_1 = some_primitive::type_id()->get_all_implementations();
    auto list_2 = some_primitive::type_id()->get_all_implementations();
    ASSERT_EQ(list_1.size(), 4);
    ASSERT_EQ(list_2.size(), 4);
    for (size_t i = 0; i < list_1.size(); i++) {
        ASSERT_EQ(list_1[i], list_2[i]) << " i = " << i;
    }
}

using PrimitiveTypeTestParams =
    std::tuple<
        some_primitive::SomeParameter,
        impl_types,
        shape_types,
        bool, // expected has_impl result
        int,  // expected count of supported impls
        int  // expected count of available impl types
    >;

class PrimitiveTypeTest : public ::testing::TestWithParam<PrimitiveTypeTestParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<PrimitiveTypeTestParams> &obj) {
        auto param_value = std::get<0>(obj.param);
        auto impl_type = std::get<1>(obj.param);
        auto shape_type = std::get<2>(obj.param);
        std::stringstream s;
        s << "v=" << static_cast<int>(param_value) << "_impl=" << impl_type << "_shape=" << shape_type;
        return s.str();
    }
};

TEST_P(PrimitiveTypeTest, has_impl_for_test) {
    auto& v = GetParam();
    auto param_value = std::get<0>(v);
    auto impl_type = std::get<1>(v);
    auto shape_type = std::get<2>(v);
    auto expected_has_impl = std::get<3>(v);
    auto expected_impls_num = std::get<4>(v);
    auto expected_impl_types_num = std::get<5>(v);

    program p(get_test_engine(), get_test_default_config(get_test_engine()));
    auto prim = std::make_shared<some_primitive>("name",  std::vector<input_info>{}, param_value);
    auto& node = p.get_or_create(prim);
    node.recalc_output_layout();

    ASSERT_EQ(some_primitive::type_id()->has_impl_for(node, impl_type, shape_type), expected_has_impl) << (int)param_value;
    if (param_value != some_primitive::SomeParameter::UNSUPPORTED_VALUE_ALL)
        ASSERT_TRUE(some_primitive::type_id()->has_impl_for(node)) << (int)param_value;
    else
        ASSERT_FALSE(some_primitive::type_id()->has_impl_for(node)) << (int)param_value;

    node.set_preferred_impl_type(impl_type);
    auto supported_impls = some_primitive::type_id()->get_supported_implementations(node);
    ASSERT_EQ(supported_impls.size(), expected_impls_num) << (int)param_value;

    auto available_types = some_primitive::type_id()->get_available_impl_types(node);
    ASSERT_EQ(available_types.size(), expected_impl_types_num) << (int)param_value;
}

INSTANTIATE_TEST_SUITE_P(smoke, PrimitiveTypeTest,
    ::testing::ValuesIn(
     std::vector<PrimitiveTypeTestParams>{
         { some_primitive::SomeParameter::SUPPORTED_VALUE_ALL, impl_types::ocl, shape_types::static_shape, true, 2, 1},
         { some_primitive::SomeParameter::SUPPORTED_VALUE_OCL_STATIC, impl_types::ocl, shape_types::static_shape, true, 1, 1},
         { some_primitive::SomeParameter::SUPPORTED_VALUE_OCL_DYNAMIC, impl_types::ocl, shape_types::static_shape, false, 1, 1},
         { some_primitive::SomeParameter::SUPPORTED_VALUE_ONEDNN_1, impl_types::ocl, shape_types::static_shape, false, 1, 1},
         { some_primitive::SomeParameter::SUPPORTED_VALUE_ONEDNN_1, impl_types::onednn, shape_types::static_shape, true, 1, 1},
         { some_primitive::SomeParameter::SUPPORTED_VALUE_ONEDNN_2, impl_types::onednn, shape_types::static_shape, true, 1, 1},
         { some_primitive::SomeParameter::SUPPORTED_VALUE_ONEDNN_1, impl_types::onednn, shape_types::dynamic_shape, false, 1, 1},
         { some_primitive::SomeParameter::UNSUPPORTED_VALUE_ALL, impl_types::ocl, shape_types::static_shape, false, 0, 0},
         { some_primitive::SomeParameter::UNSUPPORTED_VALUE_ALL, impl_types::ocl, shape_types::dynamic_shape, false, 0, 0},
    }),
    PrimitiveTypeTest::get_test_case_name);
