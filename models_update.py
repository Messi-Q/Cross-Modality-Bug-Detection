import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    """

    if layer.weight.ndimension() == 3:
        (n_out, n_in, width) = layer.weight.size()
        n = n_in * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.weight.data.fill_(1.)


# Convolutional Network
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        return x


class BytecodeNet(nn.Module):
    def __init__(self, classes_num):
        super(BytecodeNet, self).__init__()

        byetecode_dim = 256
        hiddens = 256
        bytecode_bn_dim = 128

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)

        self.bytecode_fc = nn.Linear(byetecode_dim, hiddens, bias=True)
        self.transform_fc = nn.Linear(256, hiddens, bias=True)
        self.byetecode_bn = nn.BatchNorm1d(bytecode_bn_dim)
        self.final_fc = nn.Linear(256 + hiddens, classes_num, bias=True)

        self.s_gtl = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)
        self.s_mll = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)
        self.s_bbl = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)
        self.s_bsl = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.final_fc)
        init_layer(self.transform_fc)
        init_layer(self.bytecode_fc)
        init_bn(self.byetecode_bn)

    def forward(self, input_bytecode):

        # print(input_bytecode.shape)
        (dims, seq_len) = input_bytecode.shape
        bytecode_vec = input_bytecode.view(-1, 1, seq_len)
        # print(bytecode_vec.shape)

        bytecode_vec = self.conv_block1(bytecode_vec)
        bytecode_vec = self.conv_block2(bytecode_vec)
        bytecode_vec = self.conv_block3(bytecode_vec)
        # print(bytecode_vec.shape)

        (bytecode_vec_maxpooling, _) = torch.max(bytecode_vec, dim=-1, keepdim=True)
        # print(bytecode_vec_maxpooling.shape)
        bytecode_vec_inter_rep = torch.mean(bytecode_vec_maxpooling, dim=-1)
        # print(bytecode_vec_inter_rep.shape)

        transform_bytecode = F.relu(self.transform_fc(bytecode_vec_inter_rep))
        # print(transform_bytecode.shape)

        x_combined = torch.cat([bytecode_vec_inter_rep, transform_bytecode], dim=1)
        # print(x_combined.shape)

        logit = self.final_fc(x_combined)
        student_prediction = torch.sigmoid(logit)

        return student_prediction, bytecode_vec_inter_rep, transform_bytecode


class SBFusionNet(nn.Module):
    def __init__(self, classes_num):
        super(SBFusionNet, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)

        sourcecode_dim = 256
        byetecode_dim = 256
        hiddens = 256
        sourcecode_bn_dim = 256
        bytecode_bn_dim = 256

        self.sourcecode_fc = nn.Linear(sourcecode_dim, hiddens, bias=True)
        self.byetecode_fc = nn.Linear(byetecode_dim, hiddens, bias=True)
        self.sourcecode_bn = nn.BatchNorm1d(sourcecode_bn_dim)
        self.byetecode_bn = nn.BatchNorm1d(bytecode_bn_dim)
        self.final_fc = nn.Linear(256 + hiddens, classes_num, bias=True)

        self.t_gtl = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)
        self.t_mll = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)
        self.t_bbl = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)
        self.t_bsl = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.sourcecode_fc)
        init_layer(self.byetecode_fc)
        init_bn(self.sourcecode_bn)
        init_bn(self.byetecode_bn)
        init_layer(self.final_fc)

    # teacher network going
    def forward(self, input_sourcecode, input_bytecode):
        # print(input_bytecode.shape)
        (dims, seq_len) = input_bytecode.shape
        bytecode_vec = input_bytecode.view(-1, 1, seq_len)
        # print(bytecode_vec.shape)

        bytecode_vec = self.conv_block1(bytecode_vec)
        bytecode_vec = self.conv_block2(bytecode_vec)
        bytecode_vec = self.conv_block3(bytecode_vec)
        # print(bytecode_vec.shape)
        (bytecode_vec_maxpooling, _) = torch.max(bytecode_vec, dim=-1, keepdim=True)
        # print(bytecode_vec_maxpooling.shape)
        bytecode_vec_inter_rep = torch.mean(bytecode_vec_maxpooling, dim=-1)  # teacher bytecode average pooling value
        # print(bytecode_vec_inter_rep.shape)
        sourcecode_vec = F.relu(self.sourcecode_bn(self.sourcecode_fc(input_sourcecode)))
        # print(sourcecode_vec.shape)
        sourcecode_vec = sourcecode_vec.view(len(sourcecode_vec), len(sourcecode_vec[0]), 1)
        # print(sourcecode_vec.shape)
        sourcecode_vec_inter_rep = torch.mean(sourcecode_vec, dim=-1)  # teacher sourcecode average pooling value
        # print(sourcecode_vec_inter_rep.shape)
        x_combined = torch.cat([bytecode_vec_inter_rep, sourcecode_vec_inter_rep], dim=1)
        # print(x_combined.shape)
        logit = self.final_fc(x_combined)
        teacher_prediction = torch.sigmoid(logit)  # teacher network prediction

        return teacher_prediction, bytecode_vec_inter_rep, sourcecode_vec_inter_rep
