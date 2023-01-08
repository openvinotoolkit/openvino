// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//! [include]
#include <openvino/openvino.hpp>
//! [include]

int main() {
const std::string model_path = "model.xml";
size_t batch_size = 8;

//! [ov_gna_read_model]
ov::Core core;
auto model = core.read_model(model_path);
//! [ov_gna_read_model]

//! [ov_gna_set_nc_layout]
ov::preprocess::PrePostProcessor ppp(model);
for (const auto& input : model->inputs()) {
    auto& in = ppp.input(input.get_any_name());
    in.model().set_layout(ov::Layout("N?"));
}
model = ppp.build();
//! [ov_gna_set_nc_layout]

//! [ov_gna_set_batch_size]
ov::set_batch(model, batch_size);
//! [ov_gna_set_batch_size]
}