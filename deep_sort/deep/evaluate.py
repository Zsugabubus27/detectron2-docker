import torch
import pandas as pd

# Save features, labels and frame number to a pandas dataframe
features = torch.load("features.pth")
qf = features["qf"].data.numpy()
ql = features["ql"].data.numpy()
qframe = features["qframe"].data.numpy()

qfdf = pd.DataFrame(qf)
qfdf['label'] = ql
qfdf['frameNum'] = qframe
qfdf.to_csv('vendeg_elorol_features.csv')

# features = torch.load("features.pth")
# qf = features["qf"]
# ql = features["ql"]
# gf = features["gf"]
# gl = features["gl"]

# scores = qf.mm(gf.t())
# res = scores.topk(5, dim=1)[1][:,0]
# top1correct = gl[res].eq(ql).sum().item()

# print("Acc top1:{:.3f}".format(top1correct/ql.size(0)))


