# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest


skip_commit_slider_devtest = pytest.mark.skip(
    reason="Test is used to check stability of commit_slider after development changes"
        "and does not suppose regular checks.")

