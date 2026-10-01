import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pathlib
O=str(pathlib.Path(__file__).resolve().parents[1]/"output"); R=pd.read_csv(f"{O}/tables/agreement_all24.csv"); SR=pd.read_csv(f"{O}/tables/system_rankings.csv")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300})
MODELS=["Qwen2.5-7B","Gemma-2-9B","Llama-3.1-8B","Aya-Expanse-8B"]; LANGS=["hi","bn","te","en","sw","ar"]
LN={"hi":"Hindi","bn":"Bengali","te":"Telugu","en":"English","sw":"Swahili","ar":"Arabic"}
COL={"Qwen2.5-7B":"#0072B2","Gemma-2-9B":"#009E73","Llama-3.1-8B":"#D55E00","Aya-Expanse-8B":"#CC79A7"}
def save(f,name): f.tight_layout(); f.savefig(f"{O}/figures/{name}.png"); f.savefig(f"{O}/figures/{name}.pdf"); plt.close(f)
def heat(ax,M,fmt,cmap,vmin,vmax,label,mark=None):
    im=ax.imshow(M.values,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto")
    ax.set_xticks(range(M.shape[1]),[LN[c] for c in M.columns]); ax.set_yticks(range(M.shape[0]),M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v=M.values[i,j]; t=fmt.format(v)+("*" if mark is not None and mark.values[i,j] else "")
            ax.text(j,i,t,ha="center",va="center",fontsize=8,color="white" if (v-vmin)/(vmax-vmin)>0.6 else "black")
    ax.axvline(2.5,color="black",lw=1.2); cb=plt.colorbar(im,ax=ax,fraction=0.04); cb.set_label(label)
    for s in ax.spines.values(): s.set_visible(False)
mark=R.pivot(index="model",columns="lang",values="aya_unsupported").reindex(index=MODELS,columns=LANGS)
# Fig 1 overrating
f,ax=plt.subplots(figsize=(7,2.8))
heat(ax,(R.pivot(index="model",columns="lang",values="overrating")*100).reindex(index=MODELS,columns=LANGS),"{:+.0f}","Reds",0,80,"Overrating (percentage points)",mark)
ax.set_title("Figure 1. LLM positive rate minus human positive rate (Indian languages left of line, controls right)",fontsize=9,loc="left")
save(f,"fig1_overrating_heatmap")
# Fig 2 raw vs calibrated
f,axes=plt.subplots(1,4,figsize=(9,2.9),sharey=True)
for ax,m in zip(axes,MODELS):
    d=R[R.model==m].set_index("lang").reindex(LANGS)
    for i,l in enumerate(LANGS):
        ax.plot([d.kappa[l],d.kappa_cal[l]],[i,i],color="#999999",lw=1.5,zorder=1)
    ax.scatter(d.kappa,range(6),color="white",edgecolor=COL[m],s=28,zorder=2,label="Threshold 0.5")
    ax.scatter(d.kappa_cal,range(6),color=COL[m],s=28,zorder=3,label="Calibrated (~20 queries)")
    ax.set_title(m,fontsize=9); ax.set_xlim(-0.02,0.66); ax.axhline(2.5,color="#cccccc",lw=0.8,ls="--"); ax.set_xlabel("Cohen's κ")
axes[0].set_yticks(range(6),[LN[l] for l in LANGS]); axes[0].invert_yaxis()
from matplotlib.lines import Line2D
f.legend(handles=[Line2D([],[],marker="o",ls="",mfc="white",mec="black",label="Threshold 0.5"),Line2D([],[],marker="o",ls="",color="black",label="Calibrated threshold (~20 labelled queries)")],loc="lower center",ncol=2,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,-0.02))
f.suptitle("Figure 2. Agreement before (open) and after (filled) threshold calibration",fontsize=9,x=0.01,ha="left")
f.tight_layout(rect=(0,0.06,1,1)); f.savefig(f"{O}/figures/fig2_calibration_dumbbell.png"); f.savefig(f"{O}/figures/fig2_calibration_dumbbell.pdf"); plt.close(f)
# Fig 3 AUC vs calibrated kappa
f,ax=plt.subplots(figsize=(4.6,3.6))
for m in MODELS:
    d=R[R.model==m]
    for _,r in d.iterrows():
        ax.scatter(r.auc,r.kappa_cal,color=COL[m],marker="X" if r.aya_unsupported else "o",s=45 if r.aya_unsupported else 30,edgecolor="black" if r.aya_unsupported else "none",lw=0.5)
        ax.annotate(r.lang,(r.auc,r.kappa_cal),xytext=(3,2),textcoords="offset points",fontsize=6.5,color="#444444")
    ax.scatter([],[],color=COL[m],label=m,s=30)
ax.scatter([],[],marker="X",color="#bbbbbb",edgecolor="black",label="Aya: unsupported language",s=45)
rho,p=spearmanr(R.auc,R.kappa_cal); ax.text(0.03,0.95,f"Spearman ρ = {rho:.2f} (p < 0.001), n = 24",transform=ax.transAxes,fontsize=8,va="top")
ax.set_xlabel("AUC (threshold-free discrimination)"); ax.set_ylabel("Calibrated κ"); ax.legend(fontsize=6.5,frameon=False,loc="lower right")
ax.set_title("Figure 3. Discrimination predicts calibrated agreement",fontsize=9,loc="left"); save(f,"fig3_auc_vs_calibrated_kappa")
# Fig 4 system rankings
f,(a1,a2)=plt.subplots(1,2,figsize=(9.2,3.1),gridspec_kw={"width_ratios":[1.5,1]})
heat(a1,SR.pivot(index="model",columns="lang",values="tau").reindex(index=MODELS,columns=LANGS),"{:.2f}","Blues",0,1,"Kendall's τ vs. human ranking",mark)
a1.set_title("(a) Agreement with the human ranking of 13 systems",fontsize=9,loc="left")
J=SR.merge(R[["model","lang","overrating"]],on=["model","lang"])
for m in MODELS:
    d=J[J.model==m]; a2.scatter(d.overrating*100,d.ndcg_inflation,color=COL[m],s=26,label=m)
rho,p=spearmanr(J.overrating,J.ndcg_inflation); a2.text(0.03,0.95,f"ρ = {rho:.2f} (p < 0.001)",transform=a2.transAxes,fontsize=8,va="top")
a2.axhline(0,color="#999999",lw=0.8); a2.set_xlabel("Overrating (percentage points)"); a2.set_ylabel("Mean nDCG@5: LLM − human labels")
a2.set_title("(b) Leniency inflates absolute scores",fontsize=9,loc="left"); a2.legend(fontsize=6.5,frameon=False,loc="lower right")
f.suptitle("Figure 4. System-level validity (* = language not supported by Aya)",fontsize=9,x=0.01,ha="left"); save(f,"fig4_system_rankings")
print("figures ok")
