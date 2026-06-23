// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze/ze_resource.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <type_traits>

using namespace cldnn::ze;

template <ze_resource_type resource_type>
struct ze_resource_tag {
    static constexpr ze_resource_type value = resource_type;
};

template <ocl_resource_type resource_type>
typename ocl_resource_info<resource_type>::handle_t make_ocl_handle(std::uintptr_t value) {
    using handle_t = typename ocl_resource_info<resource_type>::handle_t;
    return reinterpret_cast<handle_t>(value);
}

template <ze_resource_type resource_type>
typename ze_resource_info<resource_type>::handle_t make_ze_handle(std::uintptr_t value) {
    using handle_t = typename ze_resource_info<resource_type>::handle_t;
    if constexpr (resource_type == ze_resource_type::usm_memory) {
        return ov_ze_usm_handle{reinterpret_cast<ze_context_handle_t>(value),
                                           reinterpret_cast<void*>(value)};
    } else {
        return reinterpret_cast<handle_t>(value);
    }
}

template <ze_resource_type resource_type>
void assert_ze_handle_eq(const typename ze_resource_info<resource_type>::handle_t& expected,
                         const typename ze_resource_info<resource_type>::handle_t& actual) {
    if constexpr (resource_type == ze_resource_type::usm_memory) {
        ASSERT_EQ(expected.context, actual.context);
        ASSERT_EQ(expected.ptr, actual.ptr);
    } else {
        ASSERT_EQ(expected, actual);
    }
}

template <ze_resource_type resource_type>
struct ocl_attachment_info;

template <>
struct ocl_attachment_info<ze_resource_type::driver> {
    static constexpr ocl_resource_type ocl_type = ocl_resource_type::platform;
};

template <>
struct ocl_attachment_info<ze_resource_type::device> {
    static constexpr ocl_resource_type ocl_type = ocl_resource_type::device;
};

template <>
struct ocl_attachment_info<ze_resource_type::context> {
    static constexpr ocl_resource_type ocl_type = ocl_resource_type::context;
};

template <>
struct ocl_attachment_info<ze_resource_type::command_list> {
    static constexpr ocl_resource_type ocl_type = ocl_resource_type::command_queue;
};

template <>
struct ocl_attachment_info<ze_resource_type::usm_memory> {
    static constexpr ocl_resource_type ocl_type = ocl_resource_type::mem_object;
};

template <>
struct ocl_attachment_info<ze_resource_type::image> {
    static constexpr ocl_resource_type ocl_type = ocl_resource_type::mem_object;
};

using all_ze_resource_types = ::testing::Types<ze_resource_tag<ze_resource_type::driver>,
                                               ze_resource_tag<ze_resource_type::device>,
                                               ze_resource_tag<ze_resource_type::context>,
                                               ze_resource_tag<ze_resource_type::command_queue>,
                                               ze_resource_tag<ze_resource_type::command_list>,
                                               ze_resource_tag<ze_resource_type::module>,
                                               ze_resource_tag<ze_resource_type::kernel>,
                                               ze_resource_tag<ze_resource_type::event_pool>,
                                               ze_resource_tag<ze_resource_type::event>,
                                               ze_resource_tag<ze_resource_type::image>,
                                               ze_resource_tag<ze_resource_type::fence>,
                                               ze_resource_tag<ze_resource_type::module_build_log>,
                                               ze_resource_tag<ze_resource_type::usm_memory>>;

using ze_resource_types_with_ocl = ::testing::Types<ze_resource_tag<ze_resource_type::driver>,
                                                    ze_resource_tag<ze_resource_type::device>,
                                                    ze_resource_tag<ze_resource_type::context>,
                                                    ze_resource_tag<ze_resource_type::command_list>,
                                                    ze_resource_tag<ze_resource_type::usm_memory>,
                                                    ze_resource_tag<ze_resource_type::image>>;

template <typename resource_tag_t>
class ze_resource_all_types_test : public ::testing::Test {};

template <typename resource_tag_t>
class ze_resource_with_ocl_types_test : public ::testing::Test {};

TYPED_TEST_SUITE(ze_resource_all_types_test, all_ze_resource_types);
TYPED_TEST_SUITE(ze_resource_with_ocl_types_test, ze_resource_types_with_ocl);

TYPED_TEST(ze_resource_all_types_test, can_be_empty) {
    constexpr auto resource_type = TypeParam::value;
    using resource_t = ze_resource<resource_type>;

    resource_t resource;

    ASSERT_TRUE(resource.is_empty());
    ASSERT_EQ(resource.get_holder(), nullptr);
    ASSERT_ANY_THROW(resource.handle());
}

TYPED_TEST(ze_resource_all_types_test, can_hold_ze_handle) {
    constexpr auto resource_type = TypeParam::value;
    using resource_t = ze_resource<resource_type>;

    const auto ze_handle = make_ze_handle<resource_type>(0x100);
    resource_t resource(ze_handle, true);

    ASSERT_FALSE(resource.is_empty());
    ASSERT_NE(resource.get_holder(), nullptr);
    assert_ze_handle_eq<resource_type>(ze_handle, resource.handle());
}

TYPED_TEST(ze_resource_all_types_test, can_share_ze_handle) {
    constexpr auto resource_type = TypeParam::value;
    using resource_t = ze_resource<resource_type>;

    const auto ze_handle = make_ze_handle<resource_type>(0x200);
    std::weak_ptr<typename resource_t::ze_ocl_owner_t> weak_holder;
    resource_t resource;
    {
        auto holder = std::make_shared<typename resource_t::ze_ocl_owner_t>(ze_handle, true);
        resource = resource_t(holder);
        weak_holder = holder;
    }
    ASSERT_FALSE(weak_holder.expired());
    ASSERT_FALSE(resource.is_empty());
    ASSERT_EQ(resource.get_holder(), weak_holder.lock());

    auto copied_resource = resource;
    ASSERT_EQ(copied_resource.get_holder(), weak_holder.lock());

    resource.drop();
    ASSERT_TRUE(resource.is_empty());
    ASSERT_FALSE(weak_holder.expired());
    ASSERT_FALSE(copied_resource.is_empty());
    ASSERT_EQ(copied_resource.get_holder(), weak_holder.lock());

    copied_resource.drop();
    ASSERT_TRUE(copied_resource.is_empty());
    ASSERT_TRUE(weak_holder.expired());
}

TYPED_TEST(ze_resource_with_ocl_types_test, can_not_attach_ocl_handle_to_empty_resource) {
    constexpr auto resource_type = TypeParam::value;
    constexpr auto ocl_type = ocl_attachment_info<resource_type>::ocl_type;
    using resource_t = ze_resource<resource_type>;

    resource_t resource;

    ASSERT_FALSE(resource.template has_ocl_handle<ocl_type>());
    ASSERT_ANY_THROW(resource.template ocl_handle<ocl_type>());
    ASSERT_ANY_THROW(resource.template attach_ocl_handle<ocl_type>(make_ocl_handle<ocl_type>(0x300), true));
}

TYPED_TEST(ze_resource_with_ocl_types_test, can_attach_and_access_ocl_handle) {
    constexpr auto resource_type = TypeParam::value;
    constexpr auto ocl_type = ocl_attachment_info<resource_type>::ocl_type;
    using resource_t = ze_resource<resource_type>;

    const auto ze_handle = make_ze_handle<resource_type>(0x400);
    const auto ocl_handle = make_ocl_handle<ocl_type>(0x500);

    resource_t resource(ze_handle, true);

    ASSERT_FALSE(resource.template has_ocl_handle<ocl_type>());
    resource.template attach_ocl_handle<ocl_type>(ocl_handle, true);

    ASSERT_TRUE(resource.template has_ocl_handle<ocl_type>());
    ASSERT_EQ(resource.template ocl_handle<ocl_type>(), ocl_handle);
    ASSERT_ANY_THROW(resource.template attach_ocl_handle<ocl_type>(ocl_handle, true));
}

TYPED_TEST(ze_resource_with_ocl_types_test, can_share_ocl_handle) {
    constexpr auto resource_type = TypeParam::value;
    constexpr auto ocl_type = ocl_attachment_info<resource_type>::ocl_type;
    using resource_t = ze_resource<resource_type>;
    using ocl_owner_t = ocl_owner<ocl_type>;

    const auto ze_handle = make_ze_handle<resource_type>(0x600);
    const auto ocl_handle = make_ocl_handle<ocl_type>(0x700);

    resource_t resource(ze_handle, true);
    ASSERT_FALSE(resource.template has_ocl_handle<ocl_type>());
    auto copied_resource = resource;
    {
        ocl_owner_t owner(ocl_handle, true);
        copied_resource.template attach_ocl_handle<ocl_type>(std::move(owner));
    }
    ASSERT_TRUE(resource.template has_ocl_handle<ocl_type>());
    ASSERT_EQ(resource.template ocl_handle<ocl_type>(), ocl_handle);
    ASSERT_TRUE(copied_resource.template has_ocl_handle<ocl_type>());
    ASSERT_EQ(copied_resource.template ocl_handle<ocl_type>(), ocl_handle);
}
