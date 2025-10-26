
import argparse, time, cv2, torch, numpy as np, sys
from src.models.temporal_tcn import TCN_Tiny
LABELS=['normal','run','fall']

def preprocess(buf):
    x=np.stack(buf,0).astype(np.float32)  # [T,33,2]
    ref=x[:,:1,:]; x=x-ref
    scale=np.maximum(1e-3,np.linalg.norm(x[:,11,:]-x[:,12,:],axis=-1,keepdims=True)).mean()
    x=x/scale
    return torch.from_numpy(x).permute(2,1,0).unsqueeze(0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True)
    ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--window',type=int,default=60)
    if '--help' in sys.argv or '-h' in sys.argv:
        ap.print_help(); return
    a=ap.parse_args()
    m=TCN_Tiny(joints=33,classes=len(LABELS)); m.load_state_dict(torch.load(a.weights,map_location='cpu')); m.eval()
    cap=cv2.VideoCapture(a.device)
    if not cap.isOpened(): raise RuntimeError('Cannot open webcam')
    import mediapipe as mp
    mp_pose=mp.solutions.pose
    buf=[]; t0=time.time()
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ok,frame=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=pose.process(rgb)
            if res.pose_landmarks is None:
                lm=np.zeros((33,2),np.float32)
            else:
                lm=np.array([(p.x,p.y) for p in res.pose_landmarks.landmark],np.float32)
            buf.append(lm)
            if len(buf)>a.window: buf=buf[-a.window:]
            if len(buf)==a.window:
                x=preprocess(buf)
                with torch.no_grad(): p=m(x).softmax(1)[0].numpy()
                pred=LABELS[int(p.argmax())]; prob=float(p.max())
                cv2.putText(frame,f"{pred}:{prob:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow('PRISM USB',frame)
            if cv2.waitKey(1)&0xFF==27: break
    cap.release(); cv2.destroyAllWindows()
if __name__=='__main__': main()
