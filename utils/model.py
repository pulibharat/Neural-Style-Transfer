import torch.nn as nn
import torch


class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()


# Input
#  ↓
# Conv
#  ↓
# ReLU
#  ↓
# Pool
#  ↓
# Conv
#  ↓
# ...

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        # loads pretrained VGG weights from file
        self.vgg.load_state_dict(torch.load(vgg_path))
        # self.vgg.children() returns all layers inside Sequential.
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
#         [
#  Conv,
#  ReLU,
#  Conv,
#  ReLU,
#  Pool,
#  ...
# ]
# Because deeper VGG layers are for: classification, we only need the first 31 layers which correspond to relu4-1. We will use the output of relu4-1 as the content representation and the outputs of relu1-1, relu2-1, relu3-1, and relu4-1 as the style representation.
# so we agin created nn.sequentisl(with [:31])
# “If we remove them, why write them at all?”

# Excellent question.

# 🔵 Answer

# Because pretrained weight file expects FULL VGG architecture.

# The weights file contains weights for: Possible, BUT:

# ❌ weight loading becomes harder
# ❌ architecture mismatch risk
# ❌ easier to make mistakes

# Step 1

# Create complete VGG.

# Step 2

# Load pretrained weights correctly.

# Step 3

# Remove unnecessary layers.

        # saving the layers in a list to easily split them into blocks. We will split the first 31 layers into 4 blocks: enc_1, enc_2, enc_3, and enc_4. Each block will correspond to a specific set of layers in the VGG architecture. For example, enc_1 will contain the layers up to relu1-1, enc_2 will contain the layers up to relu2-1, and so on. This way, we can easily access the feature maps at different levels of the VGG encoder for both content and style representations.
        enc_layers = list(self.vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])

#         | Block | Learns            |
# | ----- | ----------------- |
# | enc_1 | edges/colors      |
# | enc_2 | textures          |
# | enc_3 | patterns          |
# | enc_4 | structure/content |

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            # getattr(object, attribute_name) is a built-in function in Python that returns the value of the specified attribute of an object. In this case, we are using it to access the parameters of each encoder block (enc_1, enc_2, enc_3, enc_4) in a loop. By doing this, we can set requires_grad = False for all parameters in each block without having to write separate loops for each block.
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        # for param in self.enc_1.parameters():
        #     param.requires_grad = False

        # for param in self.enc_2.parameters():
        #     param.requires_grad = False

        # for param in self.enc_3.parameters():
        #     param.requires_grad = False

        # for param in self.enc_4.parameters():
        #     param.requires_grad = False

    def forward(self, input, is_test=False):
        h1 = self.enc_1(input)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        if is_test:
            return h4
        return h1, h2, h3, h4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, input):
        return self.net(input)
