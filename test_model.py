"""
Test the multi-sensor model end-to-end.
Coordinate-conditioned decoders (NeRF-style) for spatial resolution.
"""
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, vmap
import flax.linen as nn
import optax
from scipy.interpolate import griddata
from simpeg import maps
from simpeg.potential_fields import magnetics, gravity
from discretize import TensorMesh
import time

# ============================================================
# 1. DATA (same as exercice.py)
# ============================================================
nx, ny, nz = 16, 16, 8
dx, dy, dz = 250, 250, 125
mesh = TensorMesh([np.ones(nx)*dx, np.ones(ny)*dy, np.ones(nz)*dz], origin=[0,0,-nz*dz])
n_cells = mesh.nC; cc = mesh.cell_centers

d1 = np.sqrt(((cc[:,0]-1500)/400)**2+((cc[:,1]-1500)/350)**2+((cc[:,2]+500)/150)**2)
body1 = np.exp(-d1**2*2)
d2 = np.sqrt(((cc[:,0]-2800)/350)**2+((cc[:,1]-2500)/300)**2+((cc[:,2]+700)/120)**2)
body2 = np.exp(-d2**2*2)
susceptibility_true = 0.001+body1*0.05+body2*0.02
density_true = body1*0.5+body2*0.3
Cu = 0.005+body1*0.02+body2*0.01
Ni = 0.003+body1*0.01+body2*0.025
Co = 0.001+body1*0.005+body2*0.012
gangue = 1-Cu-Ni-Co
compositions = np.stack([Cu,Ni,Co,gangue], axis=1)

V_helmert = np.array([
    [np.sqrt(3/4),-1/np.sqrt(12),-1/np.sqrt(12),-1/np.sqrt(12)],
    [0,np.sqrt(2/3),-1/np.sqrt(6),-1/np.sqrt(6)],
    [0,0,np.sqrt(1/2),-np.sqrt(1/2)]
])
def ilr_transform(comp): return np.log(comp+1e-10)@V_helmert.T
def ilr_inverse(ilr):
    c=np.exp(ilr@V_helmert); return c/c.sum(axis=1,keepdims=True)

rng_np = np.random.default_rng(42)
n_st = 20
sx = np.linspace(200,nx*dx-200,n_st); sy = np.linspace(200,ny*dy-200,n_st)
stations_mag = np.array([[x,y,80.] for x in sx for y in sy])
rec_m = magnetics.receivers.Point(stations_mag,components='tmi')
src_m = magnetics.sources.UniformBackgroundField(receiver_list=[rec_m],amplitude=55000,inclination=70,declination=0)
sim_m = magnetics.Simulation3DIntegral(mesh=mesh,survey=magnetics.Survey(src_m),chiMap=maps.IdentityMap(mesh),active_cells=np.ones(n_cells,dtype=bool),store_sensitivities='ram')
mag_obs_np = sim_m.dpred(susceptibility_true)+rng_np.normal(0,1.5,n_st**2)
G_mag = jnp.array(sim_m.G)

stations_grav = np.array([[x,y,1.] for x in sx for y in sy])
rec_g = gravity.receivers.Point(stations_grav,components='gz')
src_g = gravity.sources.SourceField(receiver_list=[rec_g])
sim_g = gravity.Simulation3DIntegral(mesh=mesh,survey=gravity.survey.Survey(source_field=src_g),rhoMap=maps.IdentityMap(mesh),active_cells=np.ones(n_cells,dtype=bool),store_sensitivities='ram')
grav_obs_np = sim_g.dpred(density_true)+rng_np.normal(0,0.02,n_st**2)
G_grav = jnp.array(sim_g.G)

surface_xx,surface_yy = np.meshgrid(mesh.cell_centers_x,mesh.cell_centers_y)
d1s = np.sqrt(((surface_xx-1500)/600)**2+((surface_yy-1500)/550)**2)
b1s = np.exp(-d1s**2*1.5)*0.4
d2s = np.sqrt(((surface_xx-2800)/550)**2+((surface_yy-2500)/500)**2)
b2s = np.exp(-d2s**2*1.5)*0.25
n_bands=10
em_bg=np.array([.30,.32,.35,.33,.30,.28,.25,.27,.30,.28])
em_fe=np.array([.15,.22,.38,.55,.68,.62,.48,.42,.38,.33])
em_cl=np.array([.40,.45,.50,.52,.58,.50,.40,.28,.18,.22])
ff=b1s+.5*b2s; fc=.5*b1s+b2s; fb=np.clip(1-ff-fc,.1,1.)
tt=fb+ff+fc; fb/=tt; ff/=tt; fc/=tt
hyper_img = np.clip(fb[:,:,None]*em_bg+ff[:,:,None]*em_fe+fc[:,:,None]*em_cl+rng_np.normal(0,.02,(ny,nx,n_bands)),0,1)

n_soil=50
sx_s=rng_np.uniform(200,nx*dx-200,n_soil); sy_s=rng_np.uniform(200,ny*dy-200,n_soil)
d1_s=np.sqrt(((sx_s-1500)/600)**2+((sy_s-1500)/550)**2); b1_s=np.exp(-d1_s**2*1.5)*0.4
d2_s=np.sqrt(((sx_s-2800)/550)**2+((sy_s-2500)/500)**2); b2_s=np.exp(-d2_s**2*1.5)*0.25
Cu_s=np.clip(30+b1_s*500+b2_s*300+rng_np.normal(0,10,n_soil),5,None)
Ni_s=np.clip(20+b1_s*200+b2_s*600+rng_np.normal(0,8,n_soil),5,None)
Co_s=np.clip(8+b1_s*80+b2_s*200+rng_np.normal(0,4,n_soil),2,None)
Fe_s=3.+b1_s*5+b2_s*3+rng_np.normal(0,.3,n_soil)
S_s=.1+b1_s*2+b2_s*1.5+rng_np.normal(0,.05,n_soil)
gxy=np.column_stack([surface_xx.ravel(),surface_yy.ravel()])
sxy=np.column_stack([sx_s,sy_s])
gc=np.zeros((nx*ny,5))
for i,v in enumerate([Cu_s,Ni_s,Co_s,Fe_s,S_s]):
    gl=griddata(sxy,v,gxy,method='linear'); gn=griddata(sxy,v,gxy,method='nearest')
    gc[:,i]=np.where(np.isnan(gl),gn,gl)
gcn=(gc-gc.min(0))/(gc.max(0)-gc.min(0)+1e-8)
geochem_2d=gcn.reshape(ny,nx,5)

hole_xy=[(1500,1500),(2800,2500),(500,500),(2000,2000),(3500,1000),(1200,1800),(2500,2200),(1800,2800),(3000,1500),(800,2500)]
forages=[]
for i,(hx,hy) in enumerate(hole_xy):
    ix=np.argmin(np.abs(mesh.cell_centers_x-hx)); iy=np.argmin(np.abs(mesh.cell_centers_y-hy))
    for iz in range(nz):
        idx=ix+iy*nx+iz*nx*ny
        cn=max(Cu[idx]*(1+rng_np.normal(0,.05)),1e-6)
        nn_=max(Ni[idx]*(1+rng_np.normal(0,.05)),1e-6)
        co=max(Co[idx]*(1+rng_np.normal(0,.05)),1e-6)
        ga=max(1-cn-nn_-co,1e-6)
        il=ilr_transform(np.array([[cn,nn_,co,ga]]))[0]
        hn=f'DH{i+1}'
        forages.append({'hole':hn,'x':mesh.cell_centers_x[ix],'y':mesh.cell_centers_y[iy],'z':mesh.cell_centers_z[iz],'ilr0':il[0],'ilr1':il[1],'ilr2':il[2],'cell_idx':idx,'is_validation':hn in {'DH4','DH7'}})
df=pd.DataFrame(forages)
df_tr=df[~df['is_validation']].reset_index(drop=True)
df_va=df[df['is_validation']].reset_index(drop=True)
print(f"Data ready: {n_st**2} mag/grav, {hyper_img.shape} hyper, {n_soil} soil, {len(df_tr)}/{len(df_va)} train/val")

# ============================================================
# 2. MODEL — Coordinate-conditioned decoders (NeRF-style)
# ============================================================

# Custom softplus with minimum gradient to prevent gradient death.
# Forward: standard softplus. Backward: gradient floor at 0.01.
@jax.custom_vjp
def safe_softplus(x):
    return jax.nn.softplus(x)
def _sp_fwd(x):
    return jax.nn.softplus(x), x
def _sp_bwd(x, g):
    return (g * jnp.maximum(jax.nn.sigmoid(x), 0.01),)
safe_softplus.defvjp(_sp_fwd, _sp_bwd)

# Custom sigmoid with minimum gradient — structurally bounds output to [0,1].
# Prevents gradient death at sigmoid extremes where derivative -> 0.
@jax.custom_vjp
def safe_sigmoid(x):
    return jax.nn.sigmoid(x)
def _ss_fwd(x):
    return jax.nn.sigmoid(x), x
def _ss_bwd(x, g):
    s = jax.nn.sigmoid(x)
    return (g * jnp.maximum(s * (1 - s), 0.01),)
safe_sigmoid.defvjp(_ss_fwd, _ss_bwd)

class Encoder(nn.Module):
    latent_dim: int=32
    @nn.compact
    def __call__(self, mg, gg, hi, gc):
        m=nn.relu(nn.Conv(16,(3,3),padding='SAME')(mg[jnp.newaxis,:,:,jnp.newaxis]))
        m=nn.relu(nn.Conv(32,(3,3),padding='SAME')(m))
        m=nn.relu(nn.Dense(64)(m.ravel()))
        g=nn.relu(nn.Conv(16,(3,3),padding='SAME')(gg[jnp.newaxis,:,:,jnp.newaxis]))
        g=nn.relu(nn.Conv(32,(3,3),padding='SAME')(g))
        g=nn.relu(nn.Dense(64)(g.ravel()))
        h=nn.relu(nn.Conv(16,(3,3),padding='SAME')(hi[jnp.newaxis,:,:,:]))
        h=nn.relu(nn.Conv(32,(3,3),padding='SAME')(h))
        h=nn.relu(nn.Dense(64)(h.ravel()))
        c=nn.relu(nn.Conv(16,(3,3),padding='SAME')(gc[jnp.newaxis,:,:,:]))
        c=nn.relu(nn.Dense(32)(c.ravel()))
        f=jnp.concatenate([m,g,h,c])
        f=nn.relu(nn.Dense(128)(f)); f=nn.relu(nn.Dense(64)(f))
        return nn.Dense(self.latent_dim)(f), nn.Dense(self.latent_dim)(f)

class DecSusc(nn.Module):
    """(z, coord_norm) → chi_local. safe_softplus prevents gradient death."""
    @nn.compact
    def __call__(self, z, coord):
        x=jnp.concatenate([z, coord])
        x=nn.relu(nn.Dense(128)(x))
        x=nn.relu(nn.Dense(64)(x))
        x=nn.relu(nn.Dense(32)(x))
        return safe_softplus(nn.Dense(1)(x).squeeze()) * 0.01

class DecDens(nn.Module):
    """(z, coord_norm) → rho_local. safe_softplus prevents gradient death."""
    @nn.compact
    def __call__(self, z, coord):
        x=jnp.concatenate([z, coord])
        x=nn.relu(nn.Dense(128)(x))
        x=nn.relu(nn.Dense(64)(x))
        x=nn.relu(nn.Dense(32)(x))
        return safe_softplus(nn.Dense(1)(x).squeeze()) * 0.1

class DecILR(nn.Module):
    """(z, coord_norm, chi_local, rho_local) → ILR (3-dim)."""
    @nn.compact
    def __call__(self, z, coord, chi_l, rho_l):
        x=jnp.concatenate([z, coord, jnp.atleast_1d(chi_l*100), jnp.atleast_1d(rho_l*10)])
        x=nn.relu(nn.Dense(64)(x))
        x=nn.relu(nn.Dense(32)(x))
        return nn.Dense(3)(x)

# ============================================================
# 3. INIT
# ============================================================
ldim=32
rng=random.PRNGKey(42); rng,*ks=random.split(rng,5)
enc=Encoder(latent_dim=ldim)
ds=DecSusc(); dd=DecDens(); di=DecILR()
dz_=jnp.zeros(ldim)
dc_=jnp.zeros(3)
params={
    'enc':enc.init(ks[0],jnp.zeros((n_st,n_st)),jnp.zeros((n_st,n_st)),jnp.zeros((ny,nx,n_bands)),jnp.zeros((ny,nx,5))),
    'ds':ds.init(ks[1],dz_,dc_),
    'dd':dd.init(ks[2],dz_,dc_),
    'di':di.init(ks[3],dz_,dc_,jnp.float32(0.),jnp.float32(0.)),
}

mg=jnp.array(mag_obs_np.reshape(n_st,n_st)); mg_n=(mg-mg.mean())/(mg.std()+1e-6)
gg=jnp.array(grav_obs_np.reshape(n_st,n_st)); gg_n=(gg-gg.mean())/(gg.std()+1e-6)
hj=jnp.array(hyper_img); gj=jnp.array(geochem_2d)
mo=jnp.array(mag_obs_np); go=jnp.array(grav_obs_np)

cm=jnp.array([nx*dx/2,ny*dy/2,-nz*dz/2]); cs=jnp.array([nx*dx/2,ny*dy/2,nz*dz/2])
all_coords_norm=jnp.array((cc-np.array(cm))/np.array(cs))
tcn=jnp.array((df_tr[['x','y','z']].values-np.array(cm))/np.array(cs))
tilr=jnp.array(df_tr[['ilr0','ilr1','ilr2']].values)
tilr_std=jnp.std(tilr,axis=0)  # per-dimension std for loss normalization
tidx=jnp.array(df_tr['cell_idx'].values,dtype=jnp.int32)
vcn=jnp.array((df_va[['x','y','z']].values-np.array(cm))/np.array(cs))
vilr=jnp.array(df_va[['ilr0','ilr1','ilr2']].values)
vidx=jnp.array(df_va['cell_idx'].values,dtype=jnp.int32)

# ============================================================
# 4. LOSS + TRAIN
# ============================================================
def loss_fn(params, rng_key, lam_recon, lam_mag, lam_grav):
    mu,lv=enc.apply(params['enc'],mg_n,gg_n,hj,gj)
    z=mu  # deterministic

    chi=vmap(lambda c:ds.apply(params['ds'],z,c))(all_coords_norm)
    rho=vmap(lambda c:dd.apply(params['dd'],z,c))(all_coords_norm)

    lm=jnp.mean((G_mag@chi-mo)**2)/jnp.var(mo)
    lg=jnp.mean((G_grav@rho-go)**2)/jnp.var(go)

    ip=vmap(lambda c,ch,rh:di.apply(params['di'],z,c,ch,rh))(tcn,chi[tidx],rho[tidx])
    lr=jnp.mean(((ip-tilr)/tilr_std)**2)  # normalized per-dimension

    # Soft upper bound penalty on rho — sum-based (not mean) to avoid dilution
    loss_rho_bound = jnp.sum(jnp.maximum(rho - 0.25, 0.0)**2)

    tot=lam_recon*lr+lam_mag*lm+lam_grav*lg + 1.0*loss_rho_bound
    return tot,{'total':tot,'recon':lr,'mag':lm,'grav':lg,
                'chi_max':jnp.max(chi),'rho_max':jnp.max(rho),
                'rho_bnd':loss_rho_bound}

n_epochs=25000
sched=optax.warmup_cosine_decay_schedule(init_value=1e-4,peak_value=2e-3,warmup_steps=500,decay_steps=n_epochs)
opt=optax.adam(sched); ost=opt.init(params)

@jax.jit
def step(params,ost,rng_key):
    (l,a),g=jax.value_and_grad(loss_fn,has_aux=True)(
        params,rng_key,10.0,1.0,0.3)
    u,ost=opt.update(g,ost,params)
    return optax.apply_updates(params,u),ost,l,a

print("\nTraining 25000 epochs (safe_softplus chi + safe_sigmoid rho)...")
t0=time.time()
for ep in range(n_epochs):
    rng,sr=random.split(rng)
    params,ost,_,aux=step(params,ost,sr)
    if ep%2500==0:
        print(f"  {ep:4d} | tot={aux['total']:.4f} rec={aux['recon']:.4f} mag={aux['mag']:.4f} "
              f"grav={aux['grav']:.4f} "
              f"chi_max={aux['chi_max']:.5f} rho_max={aux['rho_max']:.5f}")
print(f"Done in {time.time()-t0:.0f}s")

# ============================================================
# 5. EVALUATE
# ============================================================
mu,_=enc.apply(params['enc'],mg_n,gg_n,hj,gj)
chi_p=np.array(vmap(lambda c:ds.apply(params['ds'],mu,c))(all_coords_norm))
rho_p=np.array(vmap(lambda c:dd.apply(params['dd'],mu,c))(all_coords_norm))

def r2(t,p): return 1-np.sum((t-p)**2)/np.sum((t-t.mean())**2)
print(f"\nchi R²={r2(susceptibility_true,chi_p):.4f}  range: {chi_p.min():.5f}-{chi_p.max():.5f} (true: {susceptibility_true.min():.5f}-{susceptibility_true.max():.5f})")
print(f"rho R²={r2(density_true,rho_p):.4f}  range: {rho_p.min():.4f}-{rho_p.max():.4f} (true: {density_true.min():.4f}-{density_true.max():.4f})")

ilr_all=np.array(vmap(lambda c,ch,rh:di.apply(params['di'],mu,c,ch,rh))(all_coords_norm,jnp.array(chi_p),jnp.array(rho_p)))
Cu_p=ilr_inverse(ilr_all)[:,0]
print(f"Cu R²={r2(Cu,Cu_p):.4f}  corr={np.corrcoef(Cu,Cu_p)[0,1]:.4f}  range: {Cu_p.min()*100:.2f}%-{Cu_p.max()*100:.2f}% (true: {Cu.min()*100:.2f}%-{Cu.max()*100:.2f}%)")

# Val
vp=np.array(vmap(lambda c,ch,rh:di.apply(params['di'],mu,c,ch,rh))(vcn,jnp.array(chi_p)[vidx],jnp.array(rho_p)[vidx]))
print(f"Val ILR RMSE={float(np.sqrt(np.mean((vp-np.array(vilr))**2))):.4f}")

print(f"\n--- Loss breakdown ---")
for k in ['recon','mag','grav','rho_bnd']:
    print(f"  {k:10s}: {float(aux[k]):.6f}")
print(f"  z_norm    : {float(jnp.linalg.norm(mu)):.2f}")
