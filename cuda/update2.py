import re
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

def preprocess_module(module: str, args: dict, no_extern_c=False):
    '''Replaces special markers <<...>> in cuda source code with
       values provided in the args dictionary.

    '''
    def repl(match):
        contents = match.group(1)
        if contents in args:
            return str(args[contents])
        else:
            raise Exception("Failed to replace a placeholder in the source code")


    module = re.sub('<<([a-zA-Z][\w]*?)>>', repl, module)
    return SourceModule(module, no_extern_c=no_extern_c)


def load_cuda_modules(**args):
    with open("cuda/predict.cu", "r") as f:
        predict = f.read()

    with open("cuda/update2.cu", "r") as f:
        update = f.read()

    with open("cuda/rescale.cu", "r") as f:
        rescale = f.read()

    with open("cuda/resample.cu", "r") as f:
        resample = f.read()

    with open("cuda/weights_and_mean_position.cu", "r") as f:
        weights_and_mean = f.read()

    with open("cuda/permute_ref.cu", "r") as f:
        permute_ref = f.read()

    return {
        "predict": preprocess_module(predict, args, no_extern_c=True),
        "update": preprocess_module(update, args, no_extern_c=True),
        "rescale": preprocess_module(rescale, args),
        "resample": preprocess_module(resample, args),
        "weights_and_mean": preprocess_module(weights_and_mean, args),
        "permute_ref": preprocess_module(permute_ref, args)
    }
