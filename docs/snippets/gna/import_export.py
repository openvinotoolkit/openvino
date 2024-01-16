# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [import]
import openvino as ov
from io import BytesIO
#! [import]

from snippets import get_model

def main():
    model = get_model()
    blob_path = "compiled_model.blob"

    core = ov.Core()
    if "GNA" not in core.available_devices:
        return 0

    compiled_model = core.compile_model(model, device_name="GNA")

    #! [ov_gna_export]
    user_stream = compiled_model.export_model()
    with open(blob_path, "wb") as f:
        f.write(user_stream)
    #! [ov_gna_export]

    # [ov_gna_import]
    with open(blob_path, "rb") as f:
        buf = BytesIO(f.read())
        compiled_model = core.import_model(buf, device_name="GNA")
    # [ov_gna_import]
