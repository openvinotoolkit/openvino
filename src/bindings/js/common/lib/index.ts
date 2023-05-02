import Tensor from './tensor';
import Shape from './shape';

import { jsTypeByPrecisionMap } from './maps';
import { LoadModelType, IOpenVINOJSLibrary } from './types';

const modules: { [label: string]: IOpenVINOJSLibrary } = {};

export * from './types';
export default createModule;
export { Shape, Tensor, jsTypeByPrecisionMap };

function createModule(
  label: string,
  loadModel: LoadModelType,
  getVersionString: () => Promise<string>,
  getDescriptionString: () => Promise<string>
): IOpenVINOJSLibrary {
  const module = {
    Shape,
    Tensor,
    loadModel,
    getVersionString,
    getDescriptionString,
  };

  modules[label] = module;

  return module;
}
