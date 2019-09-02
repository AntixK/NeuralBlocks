import torch
import torch.nn as nn

class CompressedLinear(nn.Module):
    """
    CompressedLinear is a fully connected layer that is compressed
    using truncated SVD.

    Reference -
    [1] https://github.com/rbgirshick/py-faster-rcnn/blob/master/tools/compress_net.py
    [2] https://github.com/NervanaSystems/distiller/blob/master/jupyter/truncated_svd.ipynb
    """

    def __init__(self, weights, preserve_ratio, device="cuda"):
        super(CompressedLinear, self).__init__()
        self.weights= weights
        self.preserve_ratio = preserve_ratio
        self. U, SV = self.truncated_svd(l = int(preserve_ratio*self.weights.size(0)))

        self.linear_u = nn.Linear(self.U.size(1), self.U.size(0)).to(device)
        self.linear_u.weight.data = self.U

        self.linear_sv = nn.Linear(self.SV.size(1), self.SV.size(0)).to(device)
        self.linear_sv.weight.data = self.SV

    def truncated_svd(self, l):
        """

        :return:
        """
        U, s, V = torch.svd(self.weights, some=True)
        Ul = U[:,:l]
        sl = s[:l]
        V = V.t()
        Vl = V[:l, :]

        SV = torch.mm(torch.diag(sl), Vl)
        return Ul, SV

    def forward(self, input):
        x = self.linear_sv.forward(input)
        x = self.linear_u.forward(x)

        return x
