// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/fuse_load_store_and_convert.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::FuseLoadConvert::FuseLoadConvert() {
    MATCHER_SCOPE(FuseLoadConvert);
    auto param_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    auto load_pattern = ngraph::pattern::wrap_type<ngraph::snippets::op::Load>({param_pattern});
    auto convert_pattern = ngraph::pattern::wrap_type<ov::op::v0::Convert>({load_pattern});

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::FuseLoadConvert")
        auto& pm = m.get_pattern_value_map();
        const auto param = pm.at(param_pattern).get_node_shared_ptr();
        const auto load_shared = pm.at(load_pattern).get_node_shared_ptr();
        if (!load_shared || load_shared->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(pm.at(convert_pattern).get_node_shared_ptr());
        if (!convert || transformation_callback(convert)) {
            return false;
        }

        std::shared_ptr<ov::Node> load_convert = nullptr;
        if (const auto scalar_load = ov::as_type_ptr<ngraph::snippets::op::ScalarLoad>(load_shared)) {
            load_convert = std::make_shared<ngraph::snippets::op::ScalarLoadConvert>(param, convert->get_destination_type());
        } else if (const auto load = ov::as_type_ptr<ngraph::snippets::op::Load>(load_shared)) {
            load_convert = std::make_shared<ngraph::snippets::op::LoadConvert>(param, convert->get_destination_type(), load->get_count());
        } else {
            return false;
        }

        ngraph::copy_runtime_info(convert, load_convert);
        ngraph::replace_node(convert, load_convert);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(convert_pattern, matcher_name);
    register_matcher(m, callback);
}

ngraph::snippets::pass::FuseStoreConvert::FuseStoreConvert() {
    MATCHER_SCOPE(FuseStoreConvert);
    auto input_pattern = ngraph::pattern::any_input();
    auto convert_pattern = ngraph::pattern::wrap_type<ov::op::v0::Convert>({input_pattern});
    auto store_pattern = ngraph::pattern::wrap_type<ngraph::snippets::op::Store>({convert_pattern});

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::FuseStoreConvert")
        auto& pm = m.get_pattern_value_map();
        const auto input = pm.at(input_pattern).get_node_shared_ptr();
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(pm.at(convert_pattern).get_node_shared_ptr());
        if (!convert || convert->output(0).get_target_inputs().size() != 1 || transformation_callback(convert)) {
            return false;
        }

        std::shared_ptr<ov::Node> store_convert = nullptr;
        const auto store_shared = pm.at(store_pattern).get_node_shared_ptr();
        if (const auto scalar_store = ov::as_type_ptr<ngraph::snippets::op::ScalarStore>(store_shared)) {
            store_convert = std::make_shared<ngraph::snippets::op::ScalarStoreConvert>(input, convert->get_destination_type());
        } else if (const auto store = ov::as_type_ptr<ngraph::snippets::op::Store>(store_shared)) {
            store_convert = std::make_shared<ngraph::snippets::op::StoreConvert>(input, convert->get_destination_type(), store->get_count());
        } else {
            return false;
        }

        ngraph::copy_runtime_info(store_shared, store_convert);
        ngraph::replace_node(store_shared, store_convert);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(store_pattern, matcher_name);
    register_matcher(m, callback);
}
