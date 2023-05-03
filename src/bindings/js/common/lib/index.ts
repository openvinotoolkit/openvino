import Tensor from './tensor';
import Shape from './shape';

import { jsTypeByPrecisionMap } from './maps';
import {
  LoadModelInternalType,
  IOpenVINOJSLibrary,
  LoadModelExternalType,
  IModel,
  ModelFiles,
  ModelNameAndPath,
} from './types';

const modules: { [label: string]: IOpenVINOJSLibrary } = {};

export * from './types';
export default createModule;
export { Shape, Tensor, jsTypeByPrecisionMap };

function createModule(
  label: string,
  loadModel: LoadModelInternalType,
  getVersionString: () => Promise<string>,
  getDescriptionString: () => Promise<string>
): IOpenVINOJSLibrary {
  const module = {
    Shape,
    Tensor,
    loadModel: adaptLoadModelSignature(loadModel),
    getVersionString,
    getDescriptionString,
  };

  modules[label] = module;

  return module;
}

function adaptLoadModelSignature(loadModel: LoadModelInternalType)
: LoadModelExternalType {
  return async (
    arg: ModelFiles | ModelNameAndPath | string,
    shapeData: Shape | number[],
    layout: string
  ): Promise<IModel> => {
    let filesPaths: ModelFiles;

    if (typeof arg === 'string') {
      const pathParts = arg.split('/');
      const filename = pathParts.pop();

      arg = {
        path: pathParts.join('/'),
        modelName: filename?.replace(/(.xml)|(.bin)/, ''),
      } as ModelNameAndPath;
    }

    if (Object.prototype.hasOwnProperty.call(arg, 'path')) {
      arg = arg as ModelNameAndPath;
      filesPaths = {
        xml: `${arg.path}/${arg.modelName}.xml`,
        bin: `${arg.path}/${arg.modelName}.bin`,
      };
    }
    else filesPaths = arg as ModelFiles;

    return loadModel(filesPaths.xml, filesPaths.bin, shapeData, layout);
  };
}
