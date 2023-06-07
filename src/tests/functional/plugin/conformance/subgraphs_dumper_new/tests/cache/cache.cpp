// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "gtest/gtest.h"

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

#include "cache/cache.hpp"
#include "cache/meta.hpp"

namespace {

class ICacheUnitTest : public ::testing::Test,
                       public virtual ov::tools::subgraph_dumper::ICache {
protected:
    std::shared_ptr<ov::Model> test_model;

    void SetUp() override {
        {
            auto params = ov::ParameterVector {
                std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::PartialShape{1, 1, 1, 1})
            };
            auto convert = std::make_shared<ov::op::v0::Convert>(params.front(), ov::element::f16);
            test_model = std::make_shared<ov::Model>(convert, params);
        }
    }
};

TEST_F(ICacheUnitTest, set_serialization_dir) {
    auto test_serilization_dir("/test/dir");
    ASSERT_NO_THROW(this->set_serialization_dir(test_serilization_dir));
    ASSERT_EQ(test_serilization_dir, this->m_serialization_dir);
}

TEST_F(ICacheUnitTest, update_cache) {
    std::string test_model_path("/test/model/path.xml");
    ASSERT_NO_THROW(this->update_cache(test_model, test_model_path));
    ASSERT_NO_THROW(this->update_cache(test_model, test_model_path, true));
    ASSERT_NO_THROW(this->update_cache(test_model, test_model_path, false));
}

}  // namespace