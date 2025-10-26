import onnxruntime as ort, numpy as np, time, json

def main():
    print("[INFO] Benchmarking ONNX model")
    
    sess = ort.InferenceSession("deploy/prism_tcn_n30.onnx", providers=['CPUExecutionProvider'])
    dummy = np.random.randn(1,2,33,60).astype(np.float32)
    
    # warmup
    for _ in range(10): _=sess.run(['logits'], {'poses': dummy})
    
    N=100; t0=time.time()
    for _ in range(N): _=sess.run(['logits'], {'poses': dummy})
    dt=(time.time()-t0)/N
    
    print(f"[OK] ONNX latency per window: {dt*1000:.2f} ms")
    json.dump({"onnx_latency_ms":dt*1000}, open("runs/n30/latency_onnx.json","w"), indent=2)

if __name__=="__main__": main()
