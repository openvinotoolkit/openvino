from openvino.runtime import InferRequest
import copy
from .logging import logger


class OV_Result:
    results = None
    outputs = None

    def __init__(self, outputs):
        self.results = {}
        self.outputs = outputs

    def filter_output(self, nameset: set):
        if nameset is None or nameset.empty():
            return
        tmp_outputs = []
        for it in self.outputs:
            if it[1] in nameset:
                tmp_outputs.append(it)
        self.outputs = tmp_outputs

    def completion_callback(self, infer_request: InferRequest, index: any):
        # if index not in self.results :
        if index is None:
            return
        self.results[index] = []
        for it in self.outputs:
            # print(f"it={it}")
            self.results[index].append(copy.deepcopy(
                infer_request.get_output_tensor(it[0]).data))
        return

class ResultChecker():
    def __init__(self) -> None:
        self.target_outputs = []
    
    def set_outputs(self, namelist):
        self.target_outputs = []
        for it in namelist:
            self.target_outputs.append(it)
    
    def compare_result(self, it1, it2, target, index):
        batchsize1 = len(it1)
        batchsize2 = len(it2)
        if batchsize1 != batchsize2:
            return 1
        for j in range(batchsize1):
            data1 = it1[j]
            data2 = it2[j]
            datasize1 = len(data1)
            datasize2 = len(data2)
            if datasize1 != datasize2:
                return 2
            for k in range(datasize1):
                if data1[k] != data2[k]:
                    # logger.info(f"#### [{target}, {index}, {j}, {k}] {data1[k]} != {data2[k]}")
                    return 3
        return 0


    def compare_results(self, res1: OV_Result, res2: OV_Result):
        # total results count
        size1 = len(res1.results)
        size2 = len(res2.results)
        right_res = 0
        if size1 != size2:
            logger.info(f"compare two different results ({size1} vs {size2})")
            return False
        for i in range(size1):
            it1 = res1.results[i]
            it2 = res2.results[i]
            bEqual = True
            for target in self.target_outputs:
                ret = self.compare_result(it1[target], it2[target], target, i)
                if ret > 0 :
                    # print(f"compare result got {ret}")
                    bEqual = False
                    break
            if bEqual:
                right_res += 1
        return right_res, size1
