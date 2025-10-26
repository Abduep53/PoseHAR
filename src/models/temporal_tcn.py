
import torch, torch.nn as nn
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__(); pad=(k-1)*d//2
        self.net=nn.Sequential(nn.Conv1d(in_ch,out_ch,k,padding=pad,dilation=d), nn.ReLU(True),
                               nn.Conv1d(out_ch,out_ch,k,padding=pad,dilation=d), nn.ReLU(True))
        self.down=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x): return self.net(x)+self.down(x)
class TCN_Tiny(nn.Module):
    def __init__(self, joints=33, classes=4):
        super().__init__(); in_ch=2*joints
        self.stem=nn.Conv1d(in_ch,64,3,padding=1)
        self.b1=TemporalBlock(64,64,3,1); self.b2=TemporalBlock(64,96,3,2); self.b3=TemporalBlock(96,128,3,4)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Flatten(), nn.Linear(128,64), nn.ReLU(True), nn.Linear(64,classes))
    def forward(self,x):
        B,C,J,T=x.shape; x=x.reshape(B,C*J,T)
        x=self.stem(x); x=self.b1(x); x=self.b2(x); x=self.b3(x)
        return self.head(self.pool(x))
