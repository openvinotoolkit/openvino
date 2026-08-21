# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The GGUF frontend converts GGUF models to OpenVINO. It installs the library + headers alongside
# the other frontends (LINKABLE_FRONTEND, no SKIP_INSTALL) so direct linkers -- the llama.cpp
# ggml-openvino backend and OpenVINO GenAI -- can link openvino::frontend::gguf, and it is
# discoverable via FrontEndManager so core.read_model("model.gguf") selects it by the .gguf
# extension + GGUF magic (see supported_impl / get_front_end_data in frontend.cpp). Two ingest
# paths converge on the same op translators: a live GgufDecoder (cgraph path) and a native
# .gguf-file builder.
ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev openvino::offline_transformations)

ov_build_target_faster(openvino_gguf_frontend PCH)

# ov::parallel_for (quant dequant) needs the threading (TBB) include directories.
ov_set_threading_interface_for(openvino_gguf_frontend)
