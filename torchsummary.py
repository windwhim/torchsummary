import json
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj,torch.Tensor):
        return obj.numpy().tolist()
    else:
        return obj



def summary(model, input_size, batch_size=-1, device="cuda",verbose=False,save_file=None):

    def register_hook(module):

        def hook(module, input, output):
            
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
                    
            m_key = f'{module_idx}  {module if verbose else class_name}'
            summary[m_key] = OrderedDict()
            if isinstance(input[0], torch.Tensor):
                summary[m_key]["input_shape"] = list(input[0].size())
            elif isinstance(input[0], list):
                # 多个输入
                summary[m_key]["input_shape"] = [list(np.array(i.detach().cpu()).shape) for i in input[0]]
            summary[m_key]["input_shape"][0] = batch_size
            module_str = str(module)
            if '(' not in module_str:
                summary[m_key]["args"] = 'None'
            else:
                module_str = module_str[module_str.index("("):].replace('\n', '')
                module_str = ' '.join([word for word in module_str.split(' ') if word])
                summary[m_key]["args"] = module_str
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            else:
                summary[m_key]["trainable"] = False
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()
    widths = "{:>25} {:>80}  {:>30} {:>15}"
    print("----------------------------------------------------------------")
    line_new = widths.format("Layer (type)","Layer Args", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        output_shape_str = str(summary[layer]["output_shape"])
        line_new = widths.format(
            layer,
            summary[layer]["args"][:70]+('...' if len(summary[layer]["args"])>70 else ''),
            output_shape_str[:20]+ ('...' if len(output_shape_str)>25 else ''),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])

        if summary[layer]["trainable"] == True:
            trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    
    if save_file is not None:
        d = {'Every':summary,
             'Total':
                 {'Params':total_params,
                  'Trainable params':trainable_params,
                  'Non-trainable params':total_params - trainable_params,
                  'Input size (MB)':total_input_size,
                  'Forward/backward pass size (MB)':total_output_size,
                  'Params size (MB)':total_params_size,
                  'Estimated Total Size (MB)':total_size
                },
            }
        with open(save_file, 'w',encoding='utf-8') as f:
            json.dump(d, f,indent=2,default=default_dump)
    # return summary
