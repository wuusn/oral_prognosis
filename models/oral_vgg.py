import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn

def get_vgg16():
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 2)
    return model

def get_fcn_vgg16():
    # need to change the vgg source code, comment the flatten line in forword!
    
    # 1. LOAD PRE-TRAINED VGG16
    model = models.vgg16(pretrained=True)

    # 2. GET CONV LAYERS
    features = model.features

    # 3. GET FULLY CONNECTED LAYERS
    fcLayers = nn.Sequential(
        # stop at last layer
        *list(model.classifier.children())[:-1]
    )

    # 4. CONVERT FULLY CONNECTED LAYERS TO CONVOLUTIONAL LAYERS

    ### convert first fc layer to conv layer with 512x7x7 kernel
    fc = fcLayers[0].state_dict()
    in_ch = 512
    out_ch = fc["weight"].size(0)

    firstConv = nn.Conv2d(in_ch, out_ch, 7, 7)

    ### get the weights from the fc layer
    firstConv.load_state_dict({"weight":fc["weight"].view(out_ch, in_ch, 7, 7),
                                   "bias":fc["bias"]})

    # CREATE A LIST OF CONVS
    convList = [firstConv]

    # Similarly convert the remaining linear layers to conv layers 
    for layer in fcLayers[1:]:
        module = layer
        if isinstance(module, nn.Linear):
            # Convert the nn.Linear to nn.Conv
            fc = module.state_dict()
            in_ch = fc["weight"].size(1)
            out_ch = fc["weight"].size(0)
            conv = nn.Conv2d(in_ch, out_ch, 1, 1)

            conv.load_state_dict({"weight":fc["weight"].view(out_ch, in_ch, 1, 1),
                                                                    "bias":fc["bias"]})

            convList += [conv]
        else:
            # Append other layers such as ReLU and Dropout
            convList += [layer]

    # Set the conv layers as a nn.Sequential module
    convList += [nn.Conv2d(out_ch,2,1,1)]
    convLayers = nn.Sequential(*convList)  
    model.classifier = convLayers
    return model
