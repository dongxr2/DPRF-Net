"""Serial clean-trained official deep baselines for ESTOGU."""
from __future__ import annotations
import argparse, importlib.util, json, os, random, time
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(k,"1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import rtdl_revisiting_models as rtdl, tabm
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
REF=ROOT/"data"/"processed"/"clean_reextract.csv.gz"
FDIR=ROOT/"data"/"processed"
TMC_FILE=ROOT/"third_party"/"TMC"/"TMC ICLR"/"model.py"
KEYS=["source_file","window_id"]; MODELS=["mlp","resnet","ft_transformer","tabm","tmc"]
CONDS=["clean_reextract","vibration_missing","electrical_missing","vibration_gain_075","vibration_gain_050","vibration_gain_025","electrical_gain_075","electrical_gain_050","electrical_gain_025","vibration_drift_025rms","vibration_drift_050rms","vibration_drift_100rms","electrical_drift_025rms","electrical_drift_050rms","electrical_drift_100rms"]

def cli():
 p=argparse.ArgumentParser(); p.add_argument("--run-name",required=True); p.add_argument("--models",nargs="+",choices=MODELS,default=["mlp","resnet","ft_transformer","tabm"]); p.add_argument("--loads",nargs="+",type=int,default=[0,111,222,333,444,555]); p.add_argument("--seeds",nargs="+",type=int,default=[301,302,303,304,305]); p.add_argument("--epochs",type=int,default=80); p.add_argument("--batch-size",type=int,default=64); return p.parse_args()
def seed(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def columns(df):
 meta={"label","label_id","load","freq","source_file","window_id"}; fs=[c for c in df if c not in meta]; v=[c for c in fs if c.startswith("Vibration") or c.startswith("VibVector")]; e=[c for c in fs if c not in v]
 if (len(v),len(e))!=(41,55): raise ValueError((len(v),len(e)))
 return v,e
def tables(ref):
 need=set(CONDS)-{"vibration_missing","electrical_missing"}; expected=ref[KEYS].sort_values(KEYS).reset_index(drop=True); out={}
 for c in need:
  p=FDIR/f"{c}.csv.gz"; x=pd.read_csv(p if p.exists() else FDIR/f"{c}.csv")
  if not expected.equals(x[KEYS].sort_values(KEYS).reset_index(drop=True)): raise ValueError(f"key mismatch {c}")
  out[c]=x
 return out
def official_tmc():
 spec=importlib.util.spec_from_file_location("official_tmc",TMC_FILE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod.TMC
class Net(nn.Module):
 def __init__(self,k):
  super().__init__(); self.k=k
  if k=="mlp": self.m=rtdl.MLP(d_in=96,d_out=6,n_blocks=2,d_block=64,dropout=.1)
  elif k=="resnet": self.m=rtdl.ResNet(d_in=96,d_out=6,n_blocks=3,d_block=64,d_hidden_multiplier=2.,dropout1=.1,dropout2=.1)
  elif k=="ft_transformer": self.m=rtdl.FTTransformer(n_cont_features=96,cat_cardinalities=[],d_out=6,**rtdl.FTTransformer.get_default_kwargs(3))
  elif k=="tabm": self.m=tabm.TabM.make(n_num_features=96,cat_cardinalities=[],d_out=6)
 def forward(self,x):
  if self.k=="ft_transformer": return self.m(x,None)
  if self.k=="tabm": return self.m(x_num=x,x_cat=None)
  return self.m(x)
def batches(*a,batch,seed_value):
 return DataLoader(TensorDataset(*map(torch.from_numpy,a)),batch_size=batch,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed_value))
def train(k,xv,xe,y,s,epochs,batch,dev):
 seed(s)
 if k=="tmc":
  if dev.type!="cuda": raise RuntimeError("Official TMC requires CUDA")
  m=official_tmc()(6,2,[[41,64,64],[55,64,64]],max(1,epochs//10)).to(dev); opt=torch.optim.Adam(m.parameters(),lr=3e-4,weight_decay=1e-5); dl=batches(xv,xe,y,batch=batch,seed_value=s)
 else:
  m=Net(k).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); dl=batches(np.concatenate([xv,xe],1),y,batch=batch,seed_value=s)
 for ep in range(1,epochs+1):
  m.train()
  for z in dl:
   opt.zero_grad(set_to_none=True)
   if k=="tmc": _,_,loss=m([z[0].to(dev),z[1].to(dev)],z[2].to(dev),ep)
   else:
    logits=m(z[0].to(dev)); yy=z[1].to(dev)
    if k=="tabm": target=yy[:,None].expand(-1,logits.shape[1]); loss=F.cross_entropy(logits.flatten(0,1),target.flatten())
    else: loss=F.cross_entropy(logits,yy)
   loss.backward(); opt.step()
  if ep==1 or ep==epochs or ep%10==0: print(f"  epoch {ep}/{epochs}",flush=True)
 return m.eval()
def infer(m,k,xv,xe,dev):
 out=[]
 with torch.inference_mode():
  for i in range(0,len(xv),256):
   if k=="tmc":
    ev=m.infer([torch.from_numpy(xv[i:i+256]).to(dev),torch.from_numpy(xe[i:i+256]).to(dev)]); logits=m.DS_Combin({j:v+1 for j,v in ev.items()})
   else:
    logits=m(torch.from_numpy(np.concatenate([xv[i:i+256],xe[i:i+256]],1)).to(dev))
    if k=="tabm": logits=logits.mean(1)
   out.append(logits.argmax(1).cpu().numpy())
 return np.concatenate(out)
def main():
 a=cli(); torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 out=HERE/"results"/a.run_name
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True); ref=pd.read_csv(REF); tab=tables(ref); vc,ec=columns(ref); fs=vc+ec; rows=[]; cms={}; start_all=time.perf_counter()
 meta={"training_exposure":"clean_only","models":a.models,"loads":a.loads,"seeds":a.seeds,"epochs":a.epochs,"batch_size":a.batch_size,"conditions":CONDS,"device":str(dev),"torch_threads":torch.get_num_threads(),"rtdl_version":"0.0.2","tabm_version":"0.0.3","tmc_commit":"a3272b8746861c76a3461943b5eee51df5b5a8fe"}
 (out/"run_metadata.partial.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
 for load in a.loads:
  tr=ref[ref.load.astype(int)!=load]; te=ref[ref.load.astype(int)==load]; keys=te[KEYS].reset_index(drop=True); sv=StandardScaler().fit(tr[vc]); se=StandardScaler().fit(tr[ec]); xvtr=sv.transform(tr[vc]).astype("float32"); xetr=se.transform(tr[ec]).astype("float32"); ytr=tr.label_id.to_numpy("int64"); yt=te.label_id.to_numpy("int64"); prepared={}
  for c in CONDS:
   if c in {"vibration_missing","electrical_missing"}: continue
   frame=keys.merge(tab[c][tab[c].load.astype(int)==load],on=KEYS,how="left",validate="one_to_one")
   if frame[fs].isna().any().any(): raise ValueError(f"alignment {load} {c}")
   prepared[c]=(sv.transform(frame[vc]).astype("float32"),se.transform(frame[ec]).astype("float32"))
  cv,ce=prepared["clean_reextract"]; prepared["vibration_missing"]=(np.zeros_like(cv),ce.copy()); prepared["electrical_missing"]=(cv.copy(),np.zeros_like(ce))
  for s in a.seeds:
   for k in a.models:
    print(f"[load={load}] [seed={s}] [model={k}] training",flush=True); t=time.perf_counter(); m=train(k,xvtr,xetr,ytr,s,a.epochs,a.batch_size,dev); seconds=time.perf_counter()-t; params=sum(p.numel() for p in m.parameters())
    for c in CONDS:
     pred=infer(m,k,*prepared[c],dev); rows.append({"test_load":load,"seed":s,"model":k,"condition":c,"n_test":len(yt),"accuracy":accuracy_score(yt,pred),"macro_f1":f1_score(yt,pred,average="macro",zero_division=0),"weighted_f1":f1_score(yt,pred,average="weighted",zero_division=0),"train_time_s":seconds,"parameters":params}); cms[f"{load}|{s}|{k}|{c}"]=confusion_matrix(yt,pred,labels=range(6)).tolist()
    pd.DataFrame(rows).to_csv(out/"metrics_long.partial.csv",index=False); print(f"[load={load}] [seed={s}] [model={k}] complete ({seconds:.1f}s)",flush=True); del m; torch.cuda.empty_cache()
 metrics=pd.DataFrame(rows); metrics.to_csv(out/"metrics_long.csv",index=False); metrics.groupby(["model","condition"],as_index=False).agg(macro_f1_mean=("macro_f1","mean"),macro_f1_sd=("macro_f1","std"),accuracy_mean=("accuracy","mean"),weighted_f1_mean=("weighted_f1","mean"),train_time_s_mean=("train_time_s","mean"),parameters=("parameters","first"),n_fold_seed=("macro_f1","size")).to_csv(out/"metrics_summary.csv",index=False); (out/"confusion_matrices.json").write_text(json.dumps(cms),encoding="utf-8"); meta.update(elapsed_s=time.perf_counter()-start_all,rows=len(metrics)); (out/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); (out/"metrics_long.partial.csv").unlink(); (out/"run_metadata.partial.json").unlink(); print(json.dumps({"output":str(out),**meta},indent=2),flush=True)
if __name__=="__main__": main()
