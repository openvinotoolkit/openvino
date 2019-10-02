#distutils: language=c++
from .cimport statistics_collector_c as C


cdef class StatisticsCollector:
    def __cinit__(self,
    	        deviceName: [str, bytes],
                custom_cpu_library: [str, bytes],
                custom_cldnn: [str, bytes],
                modelFilePath: [str, bytes],
                imagesPath: [str, bytes],
                img_number: int,
                batch: int,
                progress: [str, bytes]):
        self.ppOptions._pp_size = 0
        self.ppOptions._pp_width = 0
        self.ppOptions._pp_height = 0
        self._impl = new C.StatisticsCollector(deviceName.encode(), custom_cpu_library.encode(), custom_cldnn.encode(), modelFilePath.encode(), imagesPath.encode(), img_number, batch, self.ppOptions, progress.encode())

    cpdef void collectStatisticsToIR(self, str outModelName, str output_precision):
        self._impl.collectStatisticsToIR(outModelName.encode(), output_precision.encode())

    def __dealloc__(self):
        if self._impl is not NULL:
            del self._impl
