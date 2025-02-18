// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/group_normalization_fusion.hpp"

namespace ov {
namespace test {

using GroupNormalizationFusionTestBaseValues =
    std::tuple<PartialShape,  // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               Shape,         // shape of optional instance norm gamma tensor (or empty shape if not used)
               Shape,         // shape of optional instance norm beta tensor (or empty shape if not used)
               Shape,         // shape of group norm gamma tensor
               Shape,         // shape of group norm beta tensor
               int64_t,       // number of groups
               double>;       // epsilon

using GroupNormalizationFusionSubgraphTestValues =
    std::tuple<PartialShape,   // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               Shape,          // shape of optional instance norm gamma tensor (or empty shape if not used)
               Shape,          // shape of optional instance norm beta tensor (or empty shape if not used)
               Shape,          // shape of group norm gamma tensor
               Shape,          // shape of group norm beta tensor
               int64_t,        // number of groups
               double,         // epsilon
               element::Type,  // input/output tensor element type
               std::string,    // taget device name
               AnyMap>;        // taget device properties

class GroupNormalizationFusionTestBase {
protected:
    element::Type elem_type;
    int64_t num_channels;
    bool instance_norm_gamma_present;
    bool instance_norm_beta_present;

    std::shared_ptr<op::v0::Constant> instance_norm_gamma_const;
    std::shared_ptr<op::v0::Constant> instance_norm_beta_const;
    std::shared_ptr<op::v0::Constant> group_norm_gamma_const;
    std::shared_ptr<op::v0::Constant> group_norm_beta_const;

    PartialShape data_shape;
    Shape instance_norm_gamma_shape;
    Shape instance_norm_beta_shape;
    Shape group_norm_gamma_shape;
    Shape group_norm_beta_shape;
    int64_t num_groups;
    double epsilon;

    virtual void read_test_parameters() = 0;
    void generate_weights_init_values();
    std::shared_ptr<Model> create_model();
};

class GroupNormalizationFusionSubgraphTestsF
    : public GroupNormalizationFusionTestBase,
      public SubgraphBaseStaticTest,
      public testing::WithParamInterface<GroupNormalizationFusionSubgraphTestValues> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupNormalizationFusionSubgraphTestValues>& obj);
    void run() override;

protected:
    std::string target_device_name;
    AnyMap target_configuration;

    void init_thresholds() override;
    void generate_inputs(const std::vector<Shape>& targetInputStaticShapes) override;

    void configure_device();
    void read_test_parameters();
};

template <typename... T_old_vals, typename... T_added_vals>
std::vector<std::tuple<T_old_vals..., T_added_vals...>> expand_vals(std::vector<std::tuple<T_old_vals...>> old_vals,
                                                                    std::tuple<T_added_vals...> added_vals) {
    std::vector<std::tuple<T_old_vals..., T_added_vals...>> res;
    for (const std::tuple<T_old_vals...>& t : old_vals) {
        auto new_tuple = std::tuple_cat(t, added_vals);
        res.push_back(new_tuple);
    }
    return res;
}

}  // namespace test
}  // namespace ov
