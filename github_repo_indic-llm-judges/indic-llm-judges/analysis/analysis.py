import pandas as pd, numpy as np, math, os, json
from sklearn.metrics import cohen_kappa_score as ck, roc_auc_score as auc
from scipy.stats import spearmanr, kendalltau
import pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
U=str(ROOT/"results"); OUT=str(ROOT/"output"); os.makedirs(OUT+"/figures",exist_ok=True); os.makedirs(OUT+"/tables",exist_ok=True)
M={"Qwen2.5-7B":"Qwen2_5-7B-Instruct","Gemma-2-9B":"gemma-2-9b-it","Llama-3.1-8B":"Llama-3_1-8B-Instruct","Aya-Expanse-8B":"aya-expanse-8b"}
LANGS=["hi","bn","te","en","sw","ar"]; LN={"hi":"Hindi","bn":"Bengali","te":"Telugu","en":"English","sw":"Swahili","ar":"Arabic"}
AYA_UNSUP={"bn","te","sw"}
def load(l,m,p="english",t="t2048"): return pd.read_csv(f"{U}/judgments/{l}__{M[m]}__{p}__{t}__clean.csv",dtype={"qid":str,"docid":str})
def score_col(d): return "logodds" if "logodds" in d.columns else "p_relevant"
def qidx(d): q=d.qid.unique(); return q,{x:np.where(d.qid.values==x)[0] for x in q}
def calib(d,splits=50):
    s=score_col(d); grid=np.unique(np.quantile(d[s],np.linspace(0.01,0.99,99))); qs0=sorted(d.qid.unique().tolist()); out=[]
    for k in range(splits):
        qs=qs0[:]; np.random.default_rng(k).shuffle(qs); c=set(qs[:len(qs)//5]); dc,dt=d[d.qid.isin(c)],d[~d.qid.isin(c)]
        b=max(grid,key=lambda t:ck(dc.human,(dc[s]>=t).astype(int))); out.append(ck(dt.human,(dt[s]>=b).astype(int)))
    return float(np.mean(out))
def pm(d,target=0.2,n=600,reps=200):
    rng=np.random.default_rng(0); pos=d[d.human==1]; neg=d[d.human==0]; ks=[]
    for _ in range(reps):
        s=pd.concat([pos.sample(int(n*target),replace=True,random_state=rng.integers(1e9)),neg.sample(n-int(n*target),replace=True,random_state=rng.integers(1e9))])
        ks.append(ck(s.human,(s.p_relevant>=.5).astype(int)))
    return float(np.mean(ks))
def boot(d,B=1000):
    rng=np.random.default_rng(0); q,idx=qidx(d); K=[];A=[];G=[]
    for _ in range(B):
        dd=d.iloc[np.concatenate([idx[x] for x in rng.choice(q,len(q))])]; pr=(dd.p_relevant>=.5).astype(int)
        K.append(ck(dd.human,pr)); A.append(auc(dd.human,dd.p_relevant)); G.append(pr.mean()-dd.human.mean())
    ci=lambda a:(np.percentile(a,2.5),np.percentile(a,97.5)); return ci(K),ci(A),ci(G)
rows=[]
for m in M:
    for l in LANGS:
        d=load(l,m); pr=(d.p_relevant>=.5).astype(int); (klo,khi),(alo,ahi),(glo,ghi)=boot(d)
        rows.append(dict(model=m,lang=l,language=LN[l],queries=d.qid.nunique(),pairs=len(d),human_pos=d.human.mean(),llm_pos=pr.mean(),
            overrating=pr.mean()-d.human.mean(),over_lo=glo,over_hi=ghi,kappa=ck(d.human,pr),kappa_lo=klo,kappa_hi=khi,
            kappa_cal=calib(d),kappa_pm=pm(d),auc=auc(d.human,d.p_relevant),auc_lo=alo,auc_hi=ahi,
            aya_unsupported=(m=="Aya-Expanse-8B" and l in AYA_UNSUP)))
        print(m,l,"done",flush=True)
R=pd.DataFrame(rows); R.to_csv(OUT+"/tables/agreement_all24.csv",index=False)
# ---------- stats
stats={}
r,p=spearmanr(R.auc,R.kappa_cal); stats["AUC vs calibrated kappa"]=(r,p)
r,p=spearmanr(R.overrating,R.kappa); stats["Overrating vs raw kappa"]=(r,p)
r,p=spearmanr(R.auc,R.kappa); stats["AUC vs raw kappa"]=(r,p)
# ---------- native prompt
nat=[]
for m in ["Gemma-2-9B","Llama-3.1-8B","Aya-Expanse-8B"]:
    e=load("hi",m); n=load("hi",m,"native"); mm=e.merge(n,on=["qid","docid","human"],suffixes=("_en","_nat"))
    rng=np.random.default_rng(0); q,idx=qidx(mm); dk=[];da=[];dg=[]
    for _ in range(2000):
        d=mm.iloc[np.concatenate([idx[x] for x in rng.choice(q,len(q))])]
        pe=(d.p_relevant_en>=.5).astype(int); pn=(d.p_relevant_nat>=.5).astype(int)
        dk.append(ck(d.human,pn)-ck(d.human,pe)); da.append(auc(d.human,d.p_relevant_nat)-auc(d.human,d.p_relevant_en)); dg.append(pn.mean()-pe.mean())
    pv=lambda a:2*min((np.array(a)<=0).mean(),(np.array(a)>=0).mean())
    pe=(mm.p_relevant_en>=.5).astype(int); pn=(mm.p_relevant_nat>=.5).astype(int)
    nat.append(dict(model=m,kappa_en=ck(mm.human,pe),kappa_native=ck(mm.human,pn),p_kappa=pv(dk),auc_en=auc(mm.human,mm.p_relevant_en),auc_native=auc(mm.human,mm.p_relevant_nat),p_auc=pv(da),
        overrating_en=pe.mean()-mm.human.mean(),overrating_native=pn.mean()-mm.human.mean(),p_overrating=pv(dg),kappa_cal_en=calib(e),kappa_cal_native=calib(n)))
NAT=pd.DataFrame(nat); NAT.to_csv(OUT+"/tables/native_prompt_hindi.csv",index=False)
# ---------- truncation ablation (Qwen)
tr=[]
for l in ["hi","bn","te"]:
    a=load(l,"Qwen2.5-7B",t="t512"); b=load(l,"Qwen2.5-7B")
    m2=a.merge(b,on=["qid","docid","human"],suffixes=("_512","_2048"))
    tr.append(dict(language=LN[l],kappa_512=ck(m2.human,(m2.p_relevant_512>=.5).astype(int)),kappa_2048=ck(m2.human,(m2.p_relevant_2048>=.5).astype(int)),
        auc_512=auc(m2.human,m2.p_relevant_512),auc_2048=auc(m2.human,m2.p_relevant_2048),labels_changed=((m2.p_relevant_512>=.5)!=(m2.p_relevant_2048>=.5)).mean(),
        pct_truncated_at_512=(m2.passage_tokens_2048>512).mean() if "passage_tokens_2048" in m2 else np.nan))
TR=pd.DataFrame(tr); TR.to_csv(OUT+"/tables/truncation_ablation_qwen.csv",index=False)
# ---------- system rankings
S=pd.read_csv(f"{U}/systems/system_scores.csv",dtype={"qid":str,"docid":str}); SYS=sorted(S.system.unique()); K5=5
order={k:g.sort_values(["score","docid"],ascending=[False,True]).docid.tolist() for k,g in S.groupby(["lang","system","qid"])}
disc=[1/math.log2(i+2) for i in range(60)]
def nd(docs,lab):
    g=[lab.get(x,0) for x in docs]; dcg=sum(g[i]*disc[i] for i in range(min(K5,len(g)))); ide=sorted(g,reverse=True); idcg=sum(ide[i]*disc[i] for i in range(min(K5,len(ide))))
    return dcg/idcg if idcg>0 else 0.0
def sysv(l,lab,qids): return np.array([np.mean([nd(order[(l,s,q)],lab[q]) for q in qids]) for s in SYS])
def labmap(d,c): return {q:dict(zip(g.docid,g[c].astype(int))) for q,g in d.groupby("qid")}
def tauap(j,h):
    jr=np.argsort(-j,kind="stable"); hp={s:i for i,s in enumerate(np.argsort(-h,kind="stable"))}; N=len(jr)
    return 2/(N-1)*sum(sum(hp[jr[a]]<hp[jr[i]] for a in range(i))/i for i in range(1,N))-1
sr=[]; hum={}
for l in LANGS:
    for m in M:
        d=load(l,m); d["llm"]=(d.p_relevant>=.5).astype(int); H=labmap(d,"human"); L=labmap(d,"llm"); q=sorted(H)
        h=sysv(l,H,q); j=sysv(l,L,q); hum[l]=h
        s=score_col(d); grid=np.unique(np.quantile(d[s],np.linspace(0.01,0.99,99))); tc=[];tac=[]
        for k in range(50):
            qs=q[:]; np.random.default_rng(k).shuffle(qs); c=set(qs[:len(qs)//5]); test=[x for x in q if x not in c]
            dc=d[d.qid.isin(c)]; b=max(grid,key=lambda t:ck(dc.human,(dc[s]>=t).astype(int))); d["cal"]=(d[s]>=b).astype(int)
            hh=sysv(l,H,test); jj=sysv(l,labmap(d,"cal"),test); tc.append(kendalltau(jj,hh)[0]); tac.append(tauap(jj,hh))
        sr.append(dict(model=m,lang=l,language=LN[l],tau=kendalltau(j,h)[0],tau_ap=tauap(j,h),tau_cal=np.mean(tc),tau_ap_cal=np.mean(tac),
            best_human=SYS[int(np.argmax(h))],best_llm=SYS[int(np.argmax(j))],same_best=SYS[int(np.argmax(h))]==SYS[int(np.argmax(j))],
            ndcg_human=h.mean(),ndcg_llm=j.mean(),ndcg_inflation=j.mean()-h.mean()))
SR=pd.DataFrame(sr); SR.to_csv(OUT+"/tables/system_rankings.csv",index=False)
HT=pd.DataFrame(hum,index=SYS); HT.columns=[LN[c] for c in HT.columns]; HT["Mean"]=HT.mean(axis=1); HT.sort_values("Mean",ascending=False).to_csv(OUT+"/tables/system_ndcg5_human.csv")
J=SR.merge(R[["model","lang","overrating","auc"]],on=["model","lang"])
stats["Overrating vs nDCG inflation"]=spearmanr(J.overrating,J.ndcg_inflation)
stats["AUC vs system tau"]=spearmanr(J.auc,J.tau)
# ---------- contamination
C=pd.read_csv(f"{U}/contamination/contamination_probe.csv")
C["false_match"]=(C.lang=="bn")&C.target.str.strip().str.startswith("কে")&C.completion.astype(str).str.contains("কেন")&(C.exact==1)
C["true_exact"]=C.exact.astype(bool)&~C.false_match
CS=C.groupby([C.model.str.split("/").str[-1],"lang"]).agg(probes=("exact","size"),true_exact_rate=("true_exact","mean"),mean_similarity=("similarity","mean")).reset_index()
CS.to_csv(OUT+"/tables/contamination_summary.csv",index=False)
with open(OUT+"/tables/statistics.json","w") as f: json.dump({k:{"rho":round(float(v[0]),3),"p":float(v[1])} for k,v in stats.items()},f,indent=1)

print(json.dumps({k:(round(float(v[0]),3),float(v[1])) for k,v in stats.items()},indent=1))
print(TR.round(3).to_string()); print(NAT.round(3).to_string())
print("contamination true exact overall:",C.true_exact.sum(),"/",len(C))
