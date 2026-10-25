#!/usr/bin/env python3
"""Camera-ready vector art for PhysSym, all quantitative panels derived from saved CSVs."""
import os,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from .paths import REFERENCE_DIR, ensure_output
ROOT=str(REFERENCE_DIR);O=str(ensure_output()/'figures');os.makedirs(O,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'axes.labelsize':8.5,'axes.titlesize':9.6,'axes.titleweight':'bold','xtick.labelsize':7.4,'ytick.labelsize':7.4,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
NAVY='#23364E';TEAL='#007F82';GOLD='#C67E28';BLUE='#3970AD';PURPLE='#8264A2';GRAY='#84929C';GREEN='#418B70';RED='#BB6B65';LIGHT='#EAF4F5'
P=pd.read_csv(ROOT+'/physics_results.csv');E=pd.read_csv(ROOT+'/ensemble_results.csv');T=pd.read_csv(ROOT+'/transfer_results.csv');C=pd.read_csv(ROOT+'/physics_calibration.csv')

def save(fig,name):fig.savefig(f'{O}/{name}.pdf',bbox_inches='tight',pad_inches=.08);plt.close(fig)
# 1 elegant 3-tier system figure
fig,ax=plt.subplots(figsize=(7.25,3.5));ax.set_xlim(0,12.8);ax.set_ylim(0,6.55);ax.axis('off')
def box(x,y,w,h,title,subtitle,bg,stroke):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1,rounding_size=0.14',facecolor=bg,edgecolor=stroke,lw=1.25))
 ax.text(x+.16,y+h-.29,title,color=NAVY,weight='bold',fontsize=8.5,va='top')
 ax.text(x+.16,y+.2,subtitle,color=NAVY,fontsize=7.0,va='bottom')
def arrow(a,b,rad=0,col=GRAY):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color=col,lw=1.55,connectionstyle=f'arc3,rad={rad}'))
ax.text(.2,6.16,'I   QUANTUM ORACLE',color=BLUE,weight='bold',fontsize=9)
box(.2,4.66,3.38,1.03,'Circuit + faults','Propagated Pauli classes', '#EEF3FA','#88A8C8')
box(4.77,4.66,3.38,1.03,'Symplectic oracle',r'$s_{ft}=\langle e_f,m_t\rangle_{\rm sp}$', '#EEF3FA','#88A8C8')
box(9.34,4.66,3.26,1.03,'Pauli tests','Physically executable tests', '#EEF3FA','#88A8C8')
arrow((3.71,5.19),(4.63,5.19));arrow((8.27,5.19),(9.2,5.19))
ax.text(.2,3.96,'II   PHYSICALLY CONSTRAINED DEEP LEARNING',color=TEAL,weight='bold',fontsize=9)
box(.2,2.46,3.38,1.02,'Calibration bitstrings','48 shots; node + edge moments', '#E8F6F3','#73B5A8')
box(4.77,2.46,3.38,1.02,'Factorized graph network',r'Learn $p_i$ and $c_{ij}$', '#E8F6F3','#73B5A8')
box(9.34,2.46,3.26,1.02,'Physical likelihood',r'Factorized parity likelihood', '#E8F6F3','#73B5A8')
arrow((3.71,2.97),(4.63,2.97));arrow((8.27,2.97),(9.2,2.97));arrow((11,4.5),(11,3.62),col=TEAL)
ax.text(.2,1.79,'III   CLOSED-LOOP QUANTUM DIAGNOSIS',color=GOLD,weight='bold',fontsize=9)
box(.2,.26,3.38,1.03,'Fault posterior',r'Bayesian belief update', '#FFF3E6','#D7A96C')
box(4.77,.26,3.38,1.03,'MI test selector',r'Maximum expected entropy drop', '#FFF3E6','#D7A96C')
box(9.34,.26,3.26,1.03,'Execute + update','Measure parity; revise belief', '#FFF3E6','#D7A96C')
arrow((3.71,.78),(4.63,.78));arrow((8.27,.78),(9.2,.78));arrow((11.15,2.32),(11.15,1.42),col=TEAL)
arrow((9.26,.27),(3.7,.27),rad=-.27,col=GOLD)
save(fig,'physym_architecture')

# 2 performance budget curves, each regime; oracle dark dashed
methods=['PhysicsEnsemble','PhysicsGNN','NeuralResidual','Ridge','Empirical','Oracle']
style={'PhysicsEnsemble':(TEAL,'o',2.5),'PhysicsGNN':(BLUE,'s',1.3),'NeuralResidual':(PURPLE,'^',1.25),'Ridge':(GREEN,'v',1.2),'Empirical':(GOLD,'D',1.25),'Oracle':(NAVY,'o',1.5)}
X=P[['regime','circuit','device','method','budget','acc']].copy();Y=E[['regime','circuit','device','method','budget','acc']];X=pd.concat([X,Y[Y.method=='PhysicsEnsemble']],ignore_index=True)
fig,axs=plt.subplots(1,3,figsize=(7.45,2.72),sharey=True)
for j,(ax,reg) in enumerate(zip(axs,['correlated','sparse','independent'])):
 for m in methods:
  sub=X[(X.regime==reg)&(X.method==m)].groupby('budget').acc.agg(['mean','sem'])
  col,mark,width=style[m];ax.plot(sub.index,sub['mean'],color=col,marker=mark,ms=3.4,lw=width,ls='--' if m=='Oracle' else '-',label=m)
  if m=='PhysicsEnsemble':
   ax.fill_between(sub.index,sub['mean']-1.96*sub['sem'],sub['mean']+1.96*sub['sem'],alpha=.14,color=TEAL)
 ax.set(title=['Correlated','Sparse bursts','Independent'][j],xlabel='Diagnostic executions',ylim=(0,1.01));ax.set_xticks([4,8,12,16,20]);ax.grid(axis='y',alpha=.22)
 if j==0:ax.set_ylabel('Fault top-1 accuracy')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.08),frameon=False,ncol=6,fontsize=7)
fig.tight_layout(rect=(0,.08,1,1),w_pad=.5);save(fig,'physym_budget')

# 3 8q headline bars (no arbitrary sampling; cluster CI)
fig,axs=plt.subplots(1,3,figsize=(7.5,2.65),sharey=True)
subset=['PhysicsEnsemble','NeuralResidual','Ridge','Empirical','Independent','Oracle'];cols=[TEAL,PURPLE,GREEN,GOLD,GRAY,NAVY]
for j,(ax,reg) in enumerate(zip(axs,['correlated','sparse','independent'])):
 z=X[(X.regime==reg)&(X.budget==20)].groupby('method').acc.agg(['mean','sem'])
 vals=[z.loc[m,'mean'] for m in subset]; errs=[1.96*z.loc[m,'sem'] for m in subset]
 ax.barh(np.arange(6),vals,color=cols,height=.66,xerr=errs,error_kw={'lw':.85,'capsize':2,'ecolor':NAVY})
 ax.set(title=['Correlated','Sparse bursts','Independent'][j],xlim=(0,1.02),xlabel='Accuracy');ax.invert_yaxis();ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
 ax.set_yticks(range(6));ax.set_yticklabels(['PhysSym ensemble','Unconstrained NN','Ridge calibration','Empirical parity','Independent model','Oracle channel'] if j==0 else [])
fig.tight_layout(w_pad=.4);save(fig,'physym_leaderboard')

# 4 transfer zero-shot accuracy grouped bars, correlated and sparse
fig,axs=plt.subplots(1,2,figsize=(7.05,2.73),sharey=False)
for j,reg in enumerate(['correlated','sparse']):
 ax=axs[j];sub=T[(T.regime==reg)&(T.budget==20)];bs=np.arange(3);methods=['PhysicsGNN','Empirical','Independent','Oracle']; colors=[TEAL,GOLD,GRAY,NAVY];w=.18
 for k,m in enumerate(methods):
  dat=sub[sub.method==m].groupby('n').acc.agg(['mean','sem']).loc[[8,10,12]]
  ax.bar(bs+(k-1.5)*w,dat['mean'],width=w,color=colors[k],label=m,yerr=1.96*dat['sem'],capsize=1.4,error_kw={'lw':.75})
 ax.set(title=['Correlated readout','Sparse burst readout'][j],xlabel='Qubits (trained only on 8)',ylabel='Top-1 accuracy' if j==0 else '');ax.set_xticks(bs,['8','10','12']);ax.set_ylim(0,.9 if j==1 else .65);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.07),ncol=4,frameon=False,fontsize=7.7)
fig.tight_layout(rect=(0,.07,1,1));save(fig,'physym_transfer')

# 5 calibrations: grouped MSE with log scale
fig,ax=plt.subplots(figsize=(6.55,2.74));regimes=['correlated','sparse','independent'];methods=['PhysicsGNN','NeuralResidual','Ridge','Empirical','Independent'];cc=[TEAL,PURPLE,GREEN,GOLD,GRAY]
xx=np.arange(3);w=.15
for k,m in enumerate(methods):
 dat=C[C.model==m].groupby('regime').mse.agg(['mean','sem']).loc[regimes]
 ax.bar(xx+(k-2)*w,dat['mean'],width=w,fill=True,color=cc[k],label=m)
ax.set_xticks(xx,['Correlated','Sparse bursts','Independent']);ax.set_yscale('log');ax.set_ylim(.00035,.034);ax.set(ylabel='Test parity-noise MSE (log scale)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
ax.legend(loc='upper center',ncol=5,frameon=False,fontsize=7.4,bbox_to_anchor=(.5,1.18))
fig.tight_layout(pad=.5);save(fig,'physym_calibration')

# 6 improvements heatmap vs empirical, zero-shot, actual data
R=T[T.budget==20].groupby(['n','regime','method']).acc.mean()
a=np.array([[100*(R[n,reg,'PhysicsGNN']-R[n,reg,'Empirical']) for reg in ['correlated','sparse','independent']] for n in [8,10,12]])
fig,ax=plt.subplots(figsize=(5.15,2.95));im=ax.imshow(a,cmap='BrBG',vmin=-6,vmax=6,aspect='auto')
ax.set_xticks([0,1,2],['Correlated','Sparse','Independent']);ax.set_yticks([0,1,2],['8 qubits','10 qubits','12 qubits']);
for i in range(3):
 for j in range(3):ax.text(j,i,f'{a[i,j]:+.1f} pp',ha='center',va='center',fontsize=11,fontweight='bold',color='white' if abs(a[i,j])>4 else NAVY)
ax.set_title('Out-of-distribution gains over empirical calibration')
fig.colorbar(im,ax=ax,fraction=.04,pad=.035,label='Accuracy gain (percentage points)');fig.tight_layout(pad=.6);save(fig,'physym_gainmap')
print('vector camera figures ready')
