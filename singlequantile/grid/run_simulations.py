#!/usr/bin/env python3
"""The single-quantile grid: 28 scenarios of 200 replications each (Figure 2 and Table 5 of the paper use its rows).
All fits use fixed features and convex linear-program quantile regression. The script writes
raw_results.csv, summary.csv, paired_comparisons.csv and config.json next to itself, so run a copy
(make reproduce does).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats,sparse
from scipy.optimize import linprog
import scipy

ROOT=Path(__file__).resolve().parent
TAUS=[.5,.9]; NOISES=['normal','t3','cauchy']; DELTAS=[0.,.5,1.]
KNOTS=np.array([-.5,0.,.5])
A=100.0
P=20
# A deterministic dense orthonormal DCT basis; there is no loading seed.
# Q[j,0]=1/sqrt(19), Q[j,k]=sqrt(2/19)*cos(pi*(j+.5)*k/19).
n=P-1
Q=np.empty((n,n));Q[:,0]=1/np.sqrt(n)
Q[:,1:]=np.sqrt(2/n)*np.cos(np.pi*(np.arange(n)[:,None]+.5)*np.arange(1,n)[None,:]/n)
V=np.eye(P);V[1:,1:]=Q
# Coordinate 1 is retained W; population leading PCs are columns 2 and 3.
V0=V[:,1:3]


def draw_x(rng,n):
    raw=rng.uniform(-1,1,(n,P))
    raw[:,1]*=np.sqrt(P)
    raw[:,2]*=.5*np.sqrt(P)
    x=raw@V.T
    return x, raw[:,1:3]/np.sqrt(P)


def features(z,w):
    return np.column_stack([np.ones(len(z)),z,w,
      np.maximum(z[:,0,None]-KNOTS,0),np.maximum(z[:,1],0)])


def truth(z,w):
    return 1+z[:,0]+z[:,1]+np.maximum(z[:,0],0)+w


def departure(z):
    # Nonlinear effect omitted by affine adjustment; target dictionary contains it.
    return 4*np.abs(z[:,1])-1


def target_q(z,w,delta):
    return truth(z,w)+1+z[:,0]+delta*departure(z)


def noise_sample(rng,n,name,tau):
    d={'normal':stats.norm,'t3':stats.t(df=3),'cauchy':stats.cauchy}[name]
    # Each noise has IQR one; centering makes the requested conditional tau-quantile zero.
    scale=d.ppf(.75)-d.ppf(.25)
    eps=d.rvs(size=n,random_state=rng)
    return (eps-d.ppf(tau))/scale


def centered_loss(y,pred,q,tau):
    # Directly form a difference per observation, before averaging.
    # Stable identity avoids catastrophic subtraction of two huge losses.
    # rho(y-p)-rho(y-q) = -tau*(p-q)+max(p-y,0)-max(q-y,0)
    # max difference is computed branchwise so Cauchy extremes cancel exactly.
    d=pred-q
    a=pred-y; b=q-y
    plus=np.where((a>0)&(b>0),d,np.where(a>0,a,np.where(b>0,-b,0)))
    return -tau*d+plus


def qr(X,y,tau,coef_box=None,cuts=None,return_result=False):
    n,d=X.shape
    Xs=sparse.csr_matrix(X)
    Aeq=sparse.hstack([Xs,sparse.eye(n,format='csr'),-sparse.eye(n,format='csr')],format='csr')
    cost=np.r_[np.zeros(d),np.full(n,tau/n),np.full(n,(1-tau)/n)]
    bounds=([(None,None)]*d if coef_box is None else [(-coef_box,coef_box)]*d)+[(0,None)]*(2*n)
    Aub=None if not cuts else sparse.hstack([sparse.csr_matrix(np.asarray(cuts)),sparse.csr_matrix((len(cuts),2*n))],format='csr')
    bub=None if not cuts else np.full(len(cuts),A)
    result=linprog(cost,A_eq=Aeq,b_eq=y,A_ub=Aub,b_ub=bub,bounds=bounds,method='highs',
      options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8,'threads':1})
    if not result.success:
        raise RuntimeError(f'LP failed: {result.message}')
    beta=result.x[:d]
    residual=np.max(np.abs(Aeq@result.x-y))
    answer=(beta,float(result.fun),float(residual))
    return answer+(result,) if return_result else answer


def affine_qr(X,y,tau,radius=A,gap_tolerance=1e-7):
    # The frozen design uses radius A. Cuts preserve that same Euclidean ball.
    if radius!=A: raise ValueError('Cutting-plane radius must match frozen A.')
    beta,obj,residual,result=qr(X,y,tau,return_result=True)
    initial_norm=float(np.linalg.norm(beta))
    diag=dict(fallback=0,cut_iterations=0,certified_gap=np.nan,unrestricted_norm=initial_norm)
    if initial_norm<=radius:
        return beta,obj,residual,diag
    cuts=[]
    for iteration in range(1,501):
        cuts.append(beta/np.linalg.norm(beta))
        beta,_,lp_residual,result=qr(X,y,tau,cuts=cuts,return_result=True)
        norm=float(np.linalg.norm(beta))
        feasible=beta if norm<=radius else beta*(radius/(norm*(1+4*np.finfo(float).eps)))
        r=y-X@feasible
        primal=float(np.mean(r*(tau-(r<0))))
        dual_weights=np.clip(result.eqlin.marginals,(tau-1)/len(y),tau/len(y))
        dual=float(y@dual_weights-radius*np.linalg.norm(X.T@dual_weights))
        gap=primal-dual
        if gap < -1e-7:
            raise RuntimeError(f'Invalid numerical dual gap {gap}; inspect solver scaling.')
        if gap<=gap_tolerance:
            # Construct primal residual slacks for the returned feasible coefficients.
            u=np.maximum(r,0);v=np.maximum(-r,0)
            feasible_residual=float(np.max(np.abs(X@feasible+u-v-y)))
            diag=dict(fallback=1,cut_iterations=iteration,certified_gap=max(0.,gap),unrestricted_norm=initial_norm)
            if np.linalg.norm(feasible)>radius+1e-10:raise RuntimeError('Radial feasibility failed.')
            return feasible,primal,max(lp_residual,feasible_residual),diag
    raise RuntimeError(f'Ball-constrained fit did not reach gap tolerance {gap_tolerance}; last gap {gap}.')


def pca_map(xun):
    mu=xun.mean(0)
    cov=(xun-mu).T@(xun-mu)/len(xun)
    vals,vec=np.linalg.eigh(cov)
    vh=vec[:,-2:][:,::-1]
    # Sign orientation uses fixed observed coordinate anchors, not true scores.
    # The anchors were fixed from the deterministic loading construction.
    for j in range(2):
        if vh[1,j]<0: vh[:,j]*=-1
    return mu,vh


def fit_trial(rep,name,tau,nS,nT,nEval,deltas,mode='known',N=1000,base_seed=2026,coef_box=10.):
    noise_id=NOISES.index(name); tau_id=TAUS.index(tau)
    seed=np.random.SeedSequence([base_seed,rep,noise_id,tau_id,nT,0 if mode=='known' else 1,N if mode!='known' else 0])
    rng=np.random.default_rng(seed)
    xs,zs=draw_x(rng,nS); xt,zt=draw_x(rng,nT); xe,ze=draw_x(rng,nEval)
    if mode=='pca':
        xu,_=draw_x(rng,N); mu,vh=pca_map(xu)
        zhs=(xs-mu)@vh/np.sqrt(P); zht=(xt-mu)@vh/np.sqrt(P); zhe=(xe-mu)@vh/np.sqrt(P)
        # Error diagnostic uses optimal orthogonal alignment, as in the theorem.
        u,_,vt=np.linalg.svd(vh.T@V0)
        score_error=float(np.mean(np.sum((zhe@(u@vt)-ze)**2,axis=1)))
    else:
        zhs,zht,zhe=zs,zt,ze; score_error=0.
    ws,wt,we=xs[:,0],xt[:,0],xe[:,0]
    qs,qe=truth(zs,ws),truth(ze,we)
    ns=noise_sample(rng,nS,name,tau); nt=noise_sample(rng,nT,name,tau); ne=noise_sample(rng,nEval,name,tau)
    ys=qs+ns
    Fs,Ft,Fe=features(zhs,ws),features(zht,wt),features(zhe,we)
    bs,_,resid=qr(Fs,ys,tau,coef_box=coef_box)
    qsht,qshe=Ft@bs,Fe@bs
    source_mse=float(np.mean((qshe-qe)**2))
    rows=[]
    for delta in deltas:
        qt=target_q(zt,wt,delta); qtrue=target_q(ze,we,delta)
        yt=qt+nt; ye=qtrue+ne
        bt,_,rt=qr(Ft,yt,tau,coef_box=coef_box)
        bp,_,rp=qr(np.vstack([Fs,Ft]),np.r_[ys,yt],tau,coef_box=coef_box)
        T=np.column_stack([np.ones(nT),qsht,zht,wt])
        E=np.column_stack([np.ones(nEval),qshe,zhe,we])
        ba,_,ra,diag_a=affine_qr(T,yt,tau)
        To=np.column_stack([np.ones(nT),truth(zt,wt),zt,wt])
        Eo=np.column_stack([np.ones(nEval),qe,ze,we])
        bo,_,ro,diag_o=affine_qr(To,yt,tau)
        predictions={'transfer':E@ba,'target_only':Fe@bt,'source_only':qshe,'pooled':Fe@bp,'oracle_transfer':Eo@bo}
        norms={'transfer':np.linalg.norm(ba),'oracle_transfer':np.linalg.norm(bo)}
        diagnostics={'transfer':diag_a,'oracle_transfer':diag_o}
        for method,pred in predictions.items():
            rows.append(dict(experiment=mode,rep=rep,noise=name,tau=tau,delta=delta,nS=nS,nT=nT,N=N if mode=='pca' else 0,nEval=nEval,
                method=method,mse=float(np.mean((pred-qtrue)**2)),centered_check=float(np.mean(centered_loss(ye,pred,qtrue,tau))),
                coverage=float(np.mean(ye<=pred)),source_mse=source_mse,score_mse=score_error,
                coef_norm=float(norms.get(method,np.nan)),max_equality_residual=max(resid,rt,rp,ra,ro),
                source_box_active=int(np.sum(np.isclose(np.abs(bs),coef_box,atol=1e-5))),
                target_box_active=int(np.sum(np.isclose(np.abs(bt),coef_box,atol=1e-5))),
                pooled_box_active=int(np.sum(np.isclose(np.abs(bp),coef_box,atol=1e-5))),sieve_bound=coef_box,
                affine_fallback=diagnostics.get(method,{}).get('fallback',0),
                affine_cut_iterations=diagnostics.get(method,{}).get('cut_iterations',0),
                affine_certified_gap=diagnostics.get(method,{}).get('certified_gap',np.nan),
                unrestricted_coef_norm=diagnostics.get(method,{}).get('unrestricted_norm',np.nan)))
    return rows


def summarize(df):
    keys=['experiment','noise','tau','delta','nS','nT','N','method']
    aggs={'mse_mean':('mse','mean'),'mse_sd':('mse','std'),'mse_median':('mse','median'),'replicates':('mse','size'),
          'centered_check_mean':('centered_check','mean'),'centered_check_sd':('centered_check','std'),
          'coverage_mean':('coverage','mean'),'coverage_sd':('coverage','std'),
          'source_mse_mean':('source_mse','mean'),'score_mse_mean':('score_mse','mean'),
          'max_coef_norm':('coef_norm','max'),'max_equality_residual':('max_equality_residual','max'),
          'source_box_active_total':('source_box_active','sum'),'target_box_active_total':('target_box_active','sum'),'pooled_box_active_total':('pooled_box_active','sum')}
    out=df.groupby(keys,dropna=False).agg(**aggs).reset_index()
    for metric in ['mse','centered_check','coverage']:
        out[metric+'_mcse']=out[metric+'_sd']/np.sqrt(out.replicates)
    out.to_csv(ROOT/'summary.csv',index=False)
    keys0=keys[:-1]
    wide=df.pivot(index=keys0+['rep'],columns='method',values='mse').reset_index()
    wide['transfer_minus_target']=wide.transfer-wide.target_only
    wide['transfer_minus_pooled']=wide.transfer-wide.pooled
    pairs=wide.groupby(keys0).agg(mean_difference=('transfer_minus_target','mean'),sd_difference=('transfer_minus_target','std'),
        replicates=('rep','size'),transfer_win_fraction=('transfer_minus_target',lambda s:np.mean(s<0)),
        transfer_minus_pooled=('transfer_minus_pooled','mean')).reset_index()
    pairs['paired_mcse']=pairs.sd_difference/np.sqrt(pairs.replicates)
    pairs.to_csv(ROOT/'paired_comparisons.csv',index=False)
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--replicates',type=int,default=200);parser.add_argument('--core-only',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); R=2 if args.smoke else args.replicates
    tasks=[]
    for rep in range(R):
        for noise in NOISES:
            for tau in TAUS: tasks.append((rep,noise,tau,1000,100,5000,DELTAS,'known',1000))
    if not args.core_only and not args.smoke:
        for rep in range(R):
            # Sensitivities isolate target sample size and estimation of principal scores.
            for nT in [50,200,400]: tasks.append((rep,'t3',.9,1000,nT,5000,[0.,1.],'known',1000))
            for N in [100,1000]: tasks.append((rep,'t3',.9,1000,100,5000,[0.,1.],'pca',N))
    outfile=ROOT/('smoke_raw.csv' if args.smoke else 'raw_results.csv')
    import concurrent.futures
    workers=min(4,os.cpu_count() or 1)
    config=dict(replicates=R,nS=1000,nT_core=100,nEval=5000,taus=TAUS,noises=NOISES,deltas=DELTAS,
        knots=KNOTS.tolist(),seed=2026,p=P,r=2,retained_coordinates_zero_based=[0],source_sieve_coef_box=10.,affine_ball_radius=A,
        basis_dimension=8,affine_dimension=5,loading_basis='DCT-II (19 by 19), columns in increasing frequency',primary_coefficient_box=10.,sensitivity_coefficient_box=20.,sensitivity_replicates=100,workers=workers,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version())
    if args.resume and (ROOT/'config.json').exists():
        prior=json.loads((ROOT/'config.json').read_text())
        for key in ['replicates','nS','nT_core','nEval','taus','noises','deltas','knots','seed','p','r','source_sieve_coef_box','affine_ball_radius','basis_dimension','affine_dimension']:
            if prior.get(key)!=config[key]:
                raise RuntimeError(f'Checkpoint design mismatch: {key}. Start a new directory instead.')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2))
    rows=[];start=time.time()
    if args.resume and outfile.exists():
        saved=pd.read_csv(outfile)
        for field in ['affine_fallback','affine_cut_iterations']:
            if field not in saved:saved[field]=0
        if 'affine_certified_gap' not in saved:saved['affine_certified_gap']=np.nan
        if 'unrestricted_coef_norm' not in saved:saved['unrestricted_coef_norm']=saved['coef_norm']
        keycols=['rep','noise','tau','nT','experiment','N']
        sizes=saved.groupby(keycols).size().to_dict()
        def task_key(t):
            return (t[0],t[1],t[2],t[4],t[7],t[8] if t[7]=='pca' else 0)
        expected={task_key(t):len(t[6])*5 for t in tasks}
        if any(k not in expected or n!=expected[k] for k,n in sizes.items()):
            raise RuntimeError('Checkpoint has incomplete or unexpected trial batches.')
        rows=saved.to_dict('records')
        tasks=[t for t in tasks if task_key(t) not in sizes]
        print(f'Resuming {len(tasks)} missing trial batches from {len(rows)} saved rows.',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for idx,rr in enumerate(pool.map(_worker,tasks),1):
            rows+=rr
            if idx%10==0 or idx==len(tasks):
                pd.DataFrame(rows).to_csv(outfile,index=False)
                print(f'{idx}/{len(tasks)} trials, {len(rows)} rows, {time.time()-start:.1f}s',flush=True)
    df=pd.DataFrame(rows);df.to_csv(outfile,index=False)
    if not args.smoke: summarize(df)
    print('DONE',time.time()-start,flush=True)

def _worker(args): return fit_trial(*args)
if __name__=='__main__': main()
