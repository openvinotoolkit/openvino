#inherit parent's LOCAL_PATH in Android.mk
#LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := libformat_reader
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	inference-engine/samples/common/format_reader/bmp.cpp \
	inference-engine/samples/common/format_reader/format_reader.cpp \
	inference-engine/samples/common/format_reader/MnistUbyte.cpp
#	inference-engine/samples/common/format_reader/opencv_wraper.cpp \


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/dumper \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/inference-engine/samples/common/format_reader


LOCAL_CFLAGS += -std=c++11  -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DIMPLEMENT_INFERENCE_ENGINE_API -fPIE -fvisibility=default -std=gnu++11 -D_FORTIFY_SOURCE=2 -fPIE

LOCAL_SHARED_LIBRARIES := liblog

include $(BUILD_SHARED_LIBRARY)
##########################################################################
