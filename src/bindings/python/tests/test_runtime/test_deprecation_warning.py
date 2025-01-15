# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import warnings
import numpy as np
import importlib
import openvino


def test_openvino_no_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(openvino)

        data = np.array([1, 2, 3])
        axis_vector = openvino.AxisVector(data)
        assert np.equal(axis_vector, data).all()

        assert len(w) == 0  # No warning


def test_reload_openvino_runtime():
    importlib.reload(openvino)

    def actual_test():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import openvino.runtime  # Warning is raised here
            importlib.reload(openvino.runtime)  # Second warning on reload

            assert len(w) == 2
            assert issubclass(w[-1].category, DeprecationWarning)
            assert issubclass(w[-2].category, DeprecationWarning)
            assert "The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release." in str(w[-1].message)
            assert "The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release." in str(w[-2].message)

            # Functionality is still working, no warning
            data = np.array([1, 2, 3])
            axis_vector = openvino.runtime.AxisVector(data)
            axis_vector2 = openvino.runtime.AxisVector(data)
            assert np.equal(axis_vector, data).all()
            assert np.equal(axis_vector2, data).all()

            assert len(w) == 2
    actual_test()


def test_openvino_runtime_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(openvino)

        data = np.array([1, 2, 3])
        axis_vector = openvino.runtime.AxisVector(data)  # Warning is raised here
        axis_vector2 = openvino.runtime.AxisVector(data)  # no warning
        # Functionality is still working
        assert np.equal(axis_vector, data).all()
        assert np.equal(axis_vector2, data).all()

        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release." in str(w[-1].message)


def test_openvino_runtime_warning_as():
    importlib.reload(openvino)

    def actual_test():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import openvino.runtime as rt  # Warning is raised here

            # Functionality is still working, no warning
            data = np.array([1, 2, 3])
            axis_vector = openvino.runtime.AxisVector(data)
            axis_vector2 = openvino.runtime.AxisVector(data)
            assert np.equal(axis_vector, data).all()
            assert np.equal(axis_vector2, data).all()

            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release." in str(w[-1].message)
    actual_test()


# import openvino.runtime; import openvino.runtime.utils How many warnings shall we get one or two?
def test_openvino_runtime_utils_warning():
    importlib.reload(openvino)

    def actual_test():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import openvino.runtime  # Warning is raised here
            import openvino.runtime.utils  # No warning

            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release." in str(w[-1].message)
    actual_test()
