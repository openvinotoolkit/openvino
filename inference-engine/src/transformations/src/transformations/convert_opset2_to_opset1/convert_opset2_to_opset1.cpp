// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp"
#include "transformations/convert_gelu.hpp"
#include "transformations/convert_batch_to_space.hpp"
#include "transformations/convert_space_to_batch.hpp"
#include <transformations/utils/pass_manager.hpp>
#include <memory>

bool ngraph::pass::ConvertOpSet2ToOpSet1::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto convert_gelu = ConvertGELU();
    convert_gelu.setCallback(transformation_callback);
    convert_gelu.run_on_function(f);

    auto convert_space_to_batch = ConvertSpaceToBatch();
    convert_space_to_batch.setCallback(transformation_callback);
    convert_space_to_batch.run_on_function(f);

    auto convert_batch_to_space = ConvertBatchToSpace();
    convert_batch_to_space.setCallback(transformation_callback);
    convert_batch_to_space.run_on_function(f);
    return true;
}