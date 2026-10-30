"""Reproduce population geometry views from anonymous aggregate estimates."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def save(fig,out,name):
    fig.savefig(out/f'{name}.pdf',metadata={'Author':'','Title':name.replace('_',' ')})
    fig.savefig(out/f'{name}.svg',metadata={'Creator':'Matplotlib'})
    fig.savefig(out/f'{name}.png',dpi=300)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    e=pd.read_csv(a.data/'elsa_nuclear_curve_uncertainty.csv')
    role='excess' if 'excess' in set(e.role) else 'case_minus_reference'
    e=e[e.role.eq(role)&e.marker.isin(['hba1c','chol','logferritin'])]
    if e.empty:raise ValueError('Expected cumulative excess curves in the aggregate input')
    wide=e.pivot(index='t',columns='marker',values='estimate_sd').sort_index()
    t=wide.index.to_numpy();xyz=wide[['hba1c','chol','logferritin']].to_numpy()
    fig=plt.figure(figsize=(12,4.2));grid=fig.add_gridspec(1,3,left=.07,right=.97,bottom=.2,top=.89,wspace=.4)
    ax=fig.add_subplot(grid[0,0],projection='3d');ax.set_proj_type('ortho')
    norm=Normalize(-6,4);cmap=plt.get_cmap('viridis')
    for i in range(len(t)-1):ax.plot(xyz[i:i+2,0],xyz[i:i+2,1],xyz[i:i+2,2],color=cmap(norm(t[i])),lw=2)
    for year in [-6,-4,0,4]:
        k=np.argmin(abs(t-year));ax.scatter(*xyz[k],s=22,color=cmap(norm(year)));ax.text(*xyz[k],f' {year:g} y',fontsize=8)
    ax.set(xlabel='HbA1c excess (SD)',ylabel='Cholesterol excess (SD)',zlabel='log Ferritin excess (SD)',title='Population excess trajectory')
    for j,(c1,c2,label) in enumerate([(0,1,'Glycaemia and lipid projection'),(0,2,'Glycaemia and iron projection')],start=1):
        b=fig.add_subplot(grid[0,j]);b.axhline(0,color='.85',lw=.6);b.axvline(0,color='.85',lw=.6)
        for i in range(len(t)-1):b.plot(xyz[i:i+2,c1],xyz[i:i+2,c2],color=cmap(norm(t[i])),lw=2)
        for year in [-6,-4,0,4]:
            k=np.argmin(abs(t-year));b.scatter(xyz[k,c1],xyz[k,c2],s=20,color=cmap(norm(year)));b.annotate(f'{year:g} y',(xyz[k,c1],xyz[k,c2]),xytext=(4,5),textcoords='offset points',fontsize=8)
        b.set(xlabel='HbA1c excess (SD)',ylabel=['HbA1c','Cholesterol','log Ferritin'][c2]+' excess (SD)',title=label)
    sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap);cb=fig.colorbar(sm,ax=fig.axes,orientation='horizontal',fraction=.035,pad=.15);cb.set_label('Years relative to first report')
    save(fig,a.output,'elsa_population_geometry')

    n=pd.read_csv(a.data/'nhanes_joint_centroids.csv');n=n[n.period.eq('2021-2023')]
    wide=n.pivot(index='state',columns='marker',values='standardized')
    xyz=wide[['BMI','HDL','logCRP']].to_numpy();colors=['#77848C','#BB963C','#C45532','#4C9FA7','#326D9B']
    labels=['No self-reported diagnosis; HbA1c <5.7%','No self-reported diagnosis; HbA1c 5.7–<6.5%','No self-reported diagnosis; HbA1c ≥6.5%','Self-reported diagnosis; HbA1c <6.5%','Self-reported diagnosis; HbA1c ≥6.5%']
    fig=plt.figure(figsize=(12,5.8));ax=fig.add_subplot(121,projection='3d');ax.set_proj_type('ortho')
    for i,point in enumerate(xyz):
        ax.scatter(*point,color=colors[i],s=45,label=labels[i]);ax.plot([point[0],point[0]],[point[1],point[1]],[0,point[2]],ls=':',color=colors[i],alpha=.5);ax.scatter(point[0],point[1],0,color=colors[i],s=15,alpha=.5)
        ax.text(*point,f' {i+1}',fontsize=9)
    ax.set(xlabel='BMI excess (SD)',ylabel='HDL deficit (SD)',zlabel='log CRP excess (SD)',title='Adjusted population states, 2021–23')
    ax.view_init(elev=23,azim=-55)
    b=fig.add_subplot(122);burden=pd.read_csv(a.data/'nhanes_joint_burden.csv')
    for j,period in enumerate(['2015-2016','2017-2018','2021-2023']):
        d=burden[burden.period.eq(period)].sort_values('state');yy=np.arange(5)+(j-1)*.15
        b.errorbar(d.proportion*100,yy,xerr=np.vstack([(d.proportion-d.lower)*100,(d.upper-d.proportion)*100]),fmt=['o','s','D'][j],color=['#929DA5','#4C9FA7','#C45532'][j],ms=4,capsize=2,label=period)
    b.set_yticks(np.arange(5),[f'State {i}' for i in range(1,6)]);b.invert_yaxis();b.set(xlim=(0,100),xlabel='At least two concurrent abnormalities (%)',title='Obesity, low HDL and elevated CRP');b.legend(frameon=False,loc='lower left',fontsize=8)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=1,frameon=False,fontsize=8,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.04,right=.98,top=.9,bottom=.29,wspace=.30)
    save(fig,a.output,'nhanes_population_states')
    print('Anonymous aggregate views written')

if __name__=='__main__':main()
