// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "fuse_load_store_and_convert.hpp"
#include "snippets/snippets_isa.hpp"

#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"

#include "ngraph/rt_info.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"

ov::intel_cpu::pass::FuseLoadConvert::FuseLoadConvert() {
    MATCHER_SCOPE(FuseLoadConvert);
    auto load_pattern = ngraph::pattern::wrap_type<ngraph::snippets::op::Load>();
    auto convert_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Convert>({load_pattern});

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseLoadConvert")
        auto& pm = m.get_pattern_value_map();
        const auto load_shared = pm.at(load_pattern).get_node_shared_ptr();
        if (!load_shared || load_shared->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        const auto load = std::dynamic_pointer_cast<ngraph::snippets::op::Load>(load_shared);
        if (!load)
            return false;

        const auto convert = pm.at(convert_pattern).get_node_shared_ptr();
        if (transformation_callback(convert))
            return false;

        std::shared_ptr<ngraph::Node> load_convert = nullptr;
        if (const auto convert_saturation =
                std::dynamic_pointer_cast<ngraph::snippets::op::ConvertSaturation>(convert)) {
            load_convert = std::make_shared<ov::intel_cpu::LoadConvertSaturation>(load->input_value(0),
                                                                                  convert_saturation->get_destination_type(),
                                                                                  load->get_count(), load->get_offset());
        } else if (const auto convert_truncation =
                std::dynamic_pointer_cast<ngraph::snippets::op::ConvertTruncation>(convert)) {
            load_convert = std::make_shared<ov::intel_cpu::LoadConvertTruncation>(load->input_value(0),
                                                                                  convert_truncation->get_destination_type(),
                                                                                  load->get_count(), load->get_offset());
        } else {
            throw ngraph::ngraph_error(
                "Type of Convert op is undefined. Supports only fusing Load and ConvertTruncation or ConvertSaturation ops");
        }

        if (!load_convert)
            return false;

        ngraph::copy_runtime_info(convert, load_convert);
        ngraph::replace_node(convert, load_convert);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(convert_pattern, matcher_name);
    register_matcher(m, callback);
}


ov::intel_cpu::pass::FuseStoreConvert::FuseStoreConvert() {
    MATCHER_SCOPE(FuseStoreConvert);
    auto input_pattern = ngraph::pattern::any_input();
    auto convert_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Convert>({input_pattern});
    auto store_pattern = ngraph::pattern::wrap_type<ngraph::snippets::op::Store>({convert_pattern});

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::FuseStoreConvert")
        auto& pm = m.get_pattern_value_map();
        const auto input = pm.at(input_pattern).get_node_shared_ptr();

        const auto store = std::dynamic_pointer_cast<ngraph::snippets::op::Store>(pm.at(store_pattern).get_node_shared_ptr());
        if (!store)
            return false;

        const auto convert = pm.at(convert_pattern).get_node_shared_ptr();
        if (convert->output(0).get_target_inputs().size() != 1 || transformation_callback(convert))
            return false;

        std::shared_ptr<ngraph::Node> store_convert = nullptr;
        if (const auto convert_saturation =
                std::dynamic_pointer_cast<ngraph::snippets::op::ConvertSaturation>(convert)) {
            store_convert = std::make_shared<ov::intel_cpu::StoreConvertSaturation>(input,
                                                                                    convert_saturation->get_destination_type(),
                                                                                    store->get_count(), store->get_offset());
        } else if (const auto convert_truncation =
                std::dynamic_pointer_cast<ngraph::snippets::op::ConvertTruncation>(convert)) {
            store_convert = std::make_shared<ov::intel_cpu::StoreConvertTruncation>(input,
                                                                                    convert_truncation->get_destination_type(),
                                                                                    store->get_count(), store->get_offset());
        } else {
            throw ngraph::ngraph_error(
                "Type of Convert op is undefined. Supports only fusing Store and ConvertTruncation or ConvertSaturation ops");
        }

        if (!store_convert)
            return false;

        ngraph::copy_runtime_info(store, store_convert);
        ngraph::replace_node(store, store_convert);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(store_pattern, matcher_name);
    register_matcher(m, callback);
}
