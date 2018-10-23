include $(CLEAR_VARS)

LOCAL_MODULE := classification_sample
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	inference-engine/samples/classification_sample/main.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/include/details \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/extension \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/inference-engine/samples/common \
	$(LOCAL_PATH)/inference-engine/samples/common/format_reader \
	external/gflags/android/gflags

LOCAL_CFLAGS += -std=c++14 -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -fPIE -DIMPLEMENT_INFERENCE_ENGINE_API -std=gnu++14 -D_FORTIFY_SOURCE=2 -fPIE

LOCAL_SHARED_LIBRARIES := libinference_engine liblog libformat_reader libcpu_extension
LOCAL_STATIC_LIBRARIES := libgflags

include $(BUILD_EXECUTABLE)
##########################################################################
include $(CLEAR_VARS)

LOCAL_MODULE := classification_sample_async
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	inference-engine/samples/classification_sample_async/main.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/include/details \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/extension \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/inference-engine/samples/common \
	$(LOCAL_PATH)/inference-engine/samples/common/format_reader \
	 external/gflags/android/gflags

LOCAL_CFLAGS += -std=c++14 -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -fPIE -DIMPLEMENT_INFERENCE_ENGINE_API -std=gnu++14 -D_FORTIFY_SOURCE=2 -fPIE

LOCAL_SHARED_LIBRARIES := libinference_engine liblog libformat_reader libcpu_extension
LOCAL_STATIC_LIBRARIES := libgflags

include $(BUILD_EXECUTABLE)
