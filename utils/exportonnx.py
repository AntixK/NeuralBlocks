import torch.onnx
import os

def export_onnx(model, path, batch_size, seq_len):
    pass
    # print('The model is also exported in ONNX format at {}'.
    #       format(os.path.realpath(args.onnx_export)))
    # model.eval()
    # dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    # hidden = model.init_hidden(batch_size)
    # torch.onnx.export(model, (dummy_input, hidden), path)