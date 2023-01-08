// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//! [include]
#include <fstream>
#include <openvino/openvino.hpp>
//! [include]

int main() {
const std::string model_path = "model.xml";
const std::string blob_path = "compiled_model.blob";

ov::Core core;
auto model = core.read_model(model_path);
auto compiled_model = core.compile_model(model, "GNA");

{
//! [ov_gna_export]
std::ofstream ofs(blob_path, std::ios_base::binary | std::ios::out);
compiled_model.export_model(ofs);
//! [ov_gna_export]
}
{
//! [ov_gna_import]
std::ifstream ifs(blob_path, std::ios_base::binary | std::ios_base::in);
auto compiled_model = core.import_model(ifs, "GNA");
//! [ov_gna_import]
}
}