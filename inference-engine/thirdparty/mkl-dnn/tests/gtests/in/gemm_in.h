constexpr char unused = 'x';

#if defined(FP32)
INST_TEST_CASE(TestGEMM,
    test_params{unused, 'n', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, true, mkldnn_invalid_arguments},
    test_params{unused, 't', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, true, mkldnn_invalid_arguments},
    test_params{unused, 'n', 't', 3, 2, 1, 1.0, 0.0, 3, 1, 8, true, mkldnn_invalid_arguments},
    test_params{unused, 'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, true, mkldnn_invalid_arguments},

    test_params{unused, 'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{unused, 'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{unused, 'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{unused, 't', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{unused, 'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, false},
    test_params{unused, 'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{unused, 't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{unused, 't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{unused, 'n', 'n', 2, 2, 10000, 1.0, 2.0, 2, 10000, 2, false},

    test_params{unused, 'n', 'n', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{unused, 'n', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{unused, 't', 'n', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{unused, 't', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{unused, 'n', 't', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{unused, 'n', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{unused, 't', 't', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{unused, 't', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false}
);

#else

INST_TEST_CASE(TestGEMM_expected_failures,
    test_params{'f', 'n', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, true, mkldnn_invalid_arguments},
    test_params{'f', 't', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, true, mkldnn_invalid_arguments},
    test_params{'f', 'n', 't', 3, 2, 1, 1.0, 0.0, 3, 1, 8, true, mkldnn_invalid_arguments},
    test_params{'f', 'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, true, mkldnn_invalid_arguments},

    test_params{'r', 'n', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, true, mkldnn_invalid_arguments},
    test_params{'R', 't', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, true, mkldnn_invalid_arguments},
    test_params{'r', 'n', 't', 3, 2, 1, 1.0, 0.0, 3, 1, 8, true, mkldnn_invalid_arguments},
    test_params{'R', 'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, true, mkldnn_invalid_arguments},

    test_params{'c', 'n', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, true, mkldnn_invalid_arguments},
    test_params{'C', 't', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, true, mkldnn_invalid_arguments},
    test_params{'c', 'n', 't', 3, 2, 1, 1.0, 0.0, 3, 1, 8, true, mkldnn_invalid_arguments},
    test_params{'C', 'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, true, mkldnn_invalid_arguments}
);

INST_TEST_CASE(TestGEMM_general_cases,
    /* offsetc is fixed */
    test_params{'f', 'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'f', 'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'f', 'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'f', 't', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'f', 'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, false},
    test_params{'f', 'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'f', 't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'f', 't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'f', 'n', 'n', 2, 2, 10000, 1.0, 2.0, 2, 10000, 2, false},

    /* offsetc is row */
    test_params{'r', 'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'R', 'n', 'T', 30, 20, 10, 2.0, 1.0, 120, 120, 120, false},
    test_params{'r', 'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'R', 't', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'r', 'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, false},
    test_params{'r', 'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'R', 't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'R', 't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'R', 'n', 'n', 2, 2, 10000, 1.0, 2.0, 2, 10000, 2, false},

    /* offsetc is column */
    test_params{'C', 'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'c', 'n', 'T', 30, 20, 10, 2.0, 1.0, 120, 120, 120, false},
    test_params{'c', 'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'c', 't', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, false},
    test_params{'C', 'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, false},
    test_params{'C', 'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'C', 't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'c', 't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, false},
    test_params{'c', 'n', 'n', 2, 2, 10000, 1.0, 2.0, 2, 10000, 2, false}
);

INST_TEST_CASE(TestGEMM_fractional_scales,
    /* alpha and beta have non-zero fractional part */
    test_params{'f', 'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, false},
    test_params{'F', 'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120, false},
    test_params{'f', 'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, false},
    test_params{'F', 't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, false},
    test_params{'f', 'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100, false},
    test_params{'f', 'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100, false},
    test_params{'F', 't', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100, false},
    test_params{'F', 't', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100, false},
    test_params{'f', 'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 2, 10000, 2, false},

    test_params{'r', 'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, false},
    test_params{'R', 'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120, false},
    test_params{'r', 'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, false},
    test_params{'R', 't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, false},
    test_params{'r', 'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100, false},
    test_params{'r', 'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100, false},
    test_params{'R', 't', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100, false},
    test_params{'R', 't', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100, false},
    test_params{'r', 'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 2, 10000, 2, false},

    test_params{'C', 'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, false},
    test_params{'c', 'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120, false},
    test_params{'c', 'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, false},
    test_params{'c', 't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, false},
    test_params{'C', 'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100, false},
    test_params{'C', 'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100, false},
    test_params{'C', 't', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100, false},
    test_params{'c', 't', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100, false},
    test_params{'c', 'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 2, 10000, 2, false}
);

INST_TEST_CASE(TestGEMM_heavy,
    test_params{'f', 'n', 'n', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{'f', 'n', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{'f', 't', 'n', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{'f', 't', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{'f', 'n', 't', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{'f', 'n', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},
    test_params{'f', 't', 't', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000, false},
    test_params{'f', 't', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000, false},

    test_params{'f', 'n', 'n', 2000, 2000, 2000, 2.33f, 1.66f, 2000, 2000, 2000, false},
    test_params{'f', 'n', 'n', 3000, 3000, 3000, 2.19f, 1.99f, 3000, 3000, 3000, false},
    test_params{'f', 't', 'n', 2000, 2000, 2000, 2.01f, 1.01f, 2000, 2000, 2000, false},
    test_params{'f', 't', 'n', 3000, 3000, 3000, 2.99f, 1.19f, 3000, 3000, 3000, false},
    test_params{'f', 'n', 't', 2000, 2000, 2000, 1.33f, 2.33f, 2000, 2000, 2000, false},
    test_params{'f', 'n', 't', 3000, 3000, 3000, 1.19f, 2.99f, 3000, 3000, 3000, false},
    test_params{'f', 't', 't', 2000, 2000, 2000, 1.01f, 2.01f, 2000, 2000, 2000, false},
    test_params{'f', 't', 't', 3000, 3000, 3000, 1.99f, 2.19f, 3000, 3000, 3000, false}
);
#endif
