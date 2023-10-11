#!/bin/bash
export LAYER_TESTS_INSTALL_DIR=/home/rmikhail/src/openvino/tests/layer_tests

python3 -m pip install -r ${LAYER_TESTS_INSTALL_DIR}/requirements.txt

python3 -m pytest ${LAYER_TESTS_INSTALL_DIR}/pytorch_tests -m precommit
# TensorFlow 1 Layer Tests - TF FE
# python3 -m pytest -n logical ${LAYER_TESTS_INSTALL_DIR}/tensorflow_tests/ --use_new_frontend -m precommit_tf_fe
# wait
# cat TEST-pytorch.log TEST-tf_fe.log

# TensorFlow 2 Layer Tests - TF FE
# python3 -m pytest ${LAYER_TESTS_INSTALL_DIR}/tensorflow2_keras_tests/ --use_new_frontend -m precommit_tf_fe --junitxml=${INSTALL_TEST_DIR}/TEST-tf2_fe.xml > TEST-tf2_fe.log &
# # JAX Layer Tests - TF FE
# python3 -m pytest ${LAYER_TESTS_INSTALL_DIR}/jax_tests/ -m precommit --junitxml=${INSTALL_TEST_DIR}/TEST-jax.xml > TEST-jax.log &
# # TensorFlow 1 Layer Tests - Legacy FE
# python3 -m pytest ${LAYER_TESTS_INSTALL_DIR}/tensorflow_tests/test_tf_Roll.py --ir_version=10 --junitxml=${INSTALL_TEST_DIR}/TEST-tf_Roll.xml > TEST-tf_Roll.log &
# # TensorFlow 2 Layer Tests - Legacy FE
# python3 -m pytest ${LAYER_TESTS_INSTALL_DIR}/tensorflow2_keras_tests/test_tf2_keras_activation.py --ir_version=11 --junitxml=${INSTALL_TEST_DIR}/TEST-tf2_Activation.xml -k "sigmoid" > TEST-tf2_Activation.log &
# # TensorFlow Lite Layer Tests - TFL FE
# python3 -m pytest ${LAYER_TESTS_INSTALL_DIR}/tensorflow_lite_tests/ --junitxml=${INSTALL_TEST_DIR}/TEST-tfl_fe.xml > TEST-tfl_fe.log &