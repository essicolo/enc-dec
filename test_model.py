"""
Test v3: Properly scaled regularization losses + chi_local input to ILR decoder.
Key insight: chi values are O(0.01), so raw MSE is O(0.0001) — need to normalize.
"""
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import flax.linen as nn
import optax
from simpeg import maps
from simpeg.potential_fields import magnetics
from discretize import TensorMesh

print("=== Setup ===")
nx, ny, nz = 16, 16, 8
dx, dy, dz = 250, 250, 125
mesh = TensorMesh(
    [np.ones(nx)*dx, np.ones(ny)*dy, np.ones(nz)*dz],
    origin=[0, 0, -nz*dz]
)
n_cells = mesh.nC
cc = mesh.cell_centers

d1 = np.sqrt(((cc[:,0]-1500)/400)**2 + ((cc[:,1]-1500)/350)**2 + ((cc[:,2]+500)/150)**2)
body1 = np.exp(-d1**2 * 2)
d2 = np.sqrt(((cc[:,0]-2800)/350)**2 + ((cc[:,1]-2500)/300)**2 + ((cc[:,2]+700)/120)**2)
body2 = np.exp(-d2**2 * 2)

susceptibility_true = 0.001 + body1 * 0.05 + body2 * 0.02
Cu = 0.005 + body1 * 0.02 + body2 * 0.01
Ni = 0.003 + body1 * 0.01 + body2 * 0.025
Co = 0.001 + body1 * 0.005 + body2 * 0.012
gangue = 1 - Cu - Ni - Co
compositions = np.stack([Cu, Ni, Co, gangue], axis=1)

V_helmert = np.array([
    [np.sqrt(3/4), -1/np.sqrt(12), -1/np.sqrt(12), -1/np.sqrt(12)],
    [0, np.sqrt(2/3), -1/np.sqrt(6), -1/np.sqrt(6)],
    [0, 0, np.sqrt(1/2), -np.sqrt(1/2)]
])
def ilr_transform(comp):
    return np.log(comp + 1e-10) @ V_helmert.T
def ilr_inverse(ilr):
    comp = np.exp(ilr @ V_helmert)
    return comp / comp.sum(axis=1, keepdims=True)

ilr_true = ilr_transform(compositions)

n_stations = 12
sx = np.linspace(300, nx*dx-300, n_stations)
sy = np.linspace(300, ny*dy-300, n_stations)
stations = np.array([[x, y, 50] for x in sx for y in sy])

receivers = magnetics.receivers.Point(stations, components='tmi')
source = magnetics.sources.UniformBackgroundField(
    receiver_list=[receivers], amplitude=55000, inclination=70, declination=0
)
sim = magnetics.Simulation3DIntegral(
    mesh=mesh, survey=magnetics.Survey(source), chiMap=maps.IdentityMap(mesh),
    active_cells=np.ones(n_cells, dtype=bool), store_sensitivities='ram'
)
rng_np = np.random.default_rng(42)
mag_observed = sim.dpred(susceptibility_true) + rng_np.normal(0, 1.5, len(stations))
G_jax = jnp.array(sim.G)

@jit
def forward_magnetic_jax(susceptibility):
    return G_jax @ susceptibility

# Boreholes
hole_xy = [
    (1500, 1500), (2800, 2500), (500, 500), (2000, 2000), (3500, 1000),
    (1200, 1800), (2500, 2200), (1800, 2800), (3000, 1500), (800, 2500)
]
validation_holes = {'DH4', 'DH7'}
forages = []
for i, (hx, hy) in enumerate(hole_xy):
    ix, iy = np.argmin(np.abs(mesh.cell_centers_x - hx)), np.argmin(np.abs(mesh.cell_centers_y - hy))
    for iz in range(nz):
        idx = ix + iy * nx + iz * nx * ny
        cu_n = max(Cu[idx] * (1 + rng_np.normal(0, 0.05)), 1e-6)
        ni_n = max(Ni[idx] * (1 + rng_np.normal(0, 0.05)), 1e-6)
        co_n = max(Co[idx] * (1 + rng_np.normal(0, 0.05)), 1e-6)
        ga_n = max(1 - cu_n - ni_n - co_n, 1e-6)
        ilr_n = ilr_transform(np.array([[cu_n, ni_n, co_n, ga_n]]))[0]
        hole_name = f'DH{i+1}'
        forages.append({
            'hole': hole_name, 'x': mesh.cell_centers_x[ix], 'y': mesh.cell_centers_y[iy],
            'z': mesh.cell_centers_z[iz], 'ilr0': ilr_n[0], 'ilr1': ilr_n[1], 'ilr2': ilr_n[2],
            'cell_idx': idx, 'is_validation': hole_name in validation_holes
        })
df = pd.DataFrame(forages)
df_train = df[~df['is_validation']].reset_index(drop=True)
df_val = df[df['is_validation']].reset_index(drop=True)
print(f"Train: {len(df_train)} | Val: {len(df_val)}")

print("=== Model ===")

class Encoder(nn.Module):
    latent_dim: int = 16
    @nn.compact
    def __call__(self, mag_grid):
        x = mag_grid[jnp.newaxis, :, :, jnp.newaxis]
        x = nn.Conv(16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = x.ravel()
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)
        return mu, log_var

class DecoderSusceptibility(nn.Module):
    n_cells: int
    @nn.compact
    def __call__(self, latent):
        x = nn.Dense(128)(latent)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        raw = nn.Dense(self.n_cells)(x)
        return nn.softplus(raw) * 0.01

class DecoderILR(nn.Module):
    @nn.compact
    def __call__(self, latent, coord_norm, chi_local):
        # Normalize chi_local to O(1) so network sees a meaningful feature
        chi_normalized = chi_local / 0.01
        x = jnp.concatenate([latent, jnp.atleast_1d(chi_normalized), coord_norm])
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        return nn.Dense(3)(x)

CHI_NORM = 0.01  # chi scale for normalization
CHI_REF = 0.001  # background susceptibility

def smoothness_loss(chi, nx, ny, nz):
    """Normalized smoothness: operate on chi/chi_norm so loss is O(1)."""
    chi_n = chi / CHI_NORM
    chi_3d = chi_n.reshape(nz, ny, nx)
    dx = jnp.sum((chi_3d[:, :, 1:] - chi_3d[:, :, :-1])**2)
    dy = jnp.sum((chi_3d[:, 1:, :] - chi_3d[:, :-1, :])**2)
    dz = jnp.sum((chi_3d[1:, :, :] - chi_3d[:-1, :, :])**2)
    return (dx + dy + dz) / chi.size

def smallness_loss(chi):
    """Normalized smallness: penalize deviation from background."""
    return jnp.mean(((chi - CHI_REF) / CHI_NORM)**2)

def loss_fn(params, encoder, dec_susc, dec_ilr,
            mag_grid, mag_observed,
            train_coords_norm, train_ilr, train_cell_idx,
            rng_key, lambda_physics, lambda_kl,
            lambda_smooth, lambda_small):
    mu, log_var = encoder.apply(params['encoder'], mag_grid)
    std = jnp.exp(0.5 * log_var)
    z = mu + std * random.normal(rng_key, mu.shape)

    chi_pred = dec_susc.apply(params['dec_susc'], z)
    mag_pred = forward_magnetic_jax(chi_pred)

    # Physics: normalized by data variance
    loss_physics = jnp.mean((mag_pred - mag_observed)**2) / jnp.var(mag_observed)

    # Regularization: properly scaled to O(1)
    loss_smooth = smoothness_loss(chi_pred, nx, ny, nz)
    loss_small = smallness_loss(chi_pred)

    # Reconstruction at boreholes: chi_local feeds into ILR decoder
    chi_at_train = chi_pred[train_cell_idx]
    def predict_ilr(coord, chi_local):
        return dec_ilr.apply(params['dec_ilr'], z, coord, chi_local)
    ilr_pred = vmap(predict_ilr)(train_coords_norm, chi_at_train)
    loss_recon = jnp.mean((ilr_pred - train_ilr)**2)

    loss_kl = -0.5 * jnp.mean(1 + log_var - mu**2 - jnp.exp(log_var))

    loss_total = (loss_recon + lambda_kl * loss_kl + lambda_physics * loss_physics
                  + lambda_smooth * loss_smooth + lambda_small * loss_small)

    return loss_total, {
        'total': loss_total, 'recon': loss_recon, 'kl': loss_kl,
        'physics': loss_physics, 'smooth': loss_smooth, 'small': loss_small,
        'chi_mean': jnp.mean(chi_pred), 'chi_max': jnp.max(chi_pred)
    }

rng = random.PRNGKey(42)
rng, *init_rngs = random.split(rng, 4)

latent_dim = 16
encoder = Encoder(latent_dim=latent_dim)
dec_susc = DecoderSusceptibility(n_cells=n_cells)
dec_ilr = DecoderILR()

params = {
    'encoder': encoder.init(init_rngs[0], jnp.zeros((n_stations, n_stations))),
    'dec_susc': dec_susc.init(init_rngs[1], jnp.zeros(latent_dim)),
    'dec_ilr': dec_ilr.init(init_rngs[2], jnp.zeros(latent_dim), jnp.zeros(3), jnp.float32(0.0))
}

mag_grid = jnp.array(mag_observed.reshape(n_stations, n_stations))
mag_grid_norm = (mag_grid - mag_grid.mean()) / (mag_grid.std() + 1e-6)
mag_obs_jax = jnp.array(mag_observed)

coord_mean = jnp.array([nx*dx/2, ny*dy/2, -nz*dz/2])
coord_std = jnp.array([nx*dx/2, ny*dy/2, nz*dz/2])
train_coords_norm = jnp.array((df_train[['x', 'y', 'z']].values - np.array(coord_mean)) / np.array(coord_std))
train_ilr = jnp.array(df_train[['ilr0', 'ilr1', 'ilr2']].values)
train_cell_idx = jnp.array(df_train['cell_idx'].values, dtype=jnp.int32)
val_coords_norm = jnp.array((df_val[['x', 'y', 'z']].values - np.array(coord_mean)) / np.array(coord_std))
val_ilr = jnp.array(df_val[['ilr0', 'ilr1', 'ilr2']].values)
val_cell_idx = jnp.array(df_val['cell_idx'].values, dtype=jnp.int32)

print("=== Training (5000 epochs) ===")
n_epochs = 5000
schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4, peak_value=1e-3, warmup_steps=200, decay_steps=n_epochs
)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(params)

kl_warmup = 1000
kl_target = 0.01

@jit
def train_step(params, opt_state, rng_key, lambda_kl):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, encoder, dec_susc, dec_ilr,
        mag_grid_norm, mag_obs_jax,
        train_coords_norm, train_ilr, train_cell_idx,
        rng_key,
        lambda_physics=1.0,
        lambda_kl=lambda_kl,
        lambda_smooth=0.1,   # now properly scaled — this is meaningful
        lambda_small=0.05    # penalize deviation from background
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

for epoch in range(n_epochs):
    rng, step_rng = random.split(rng)
    lkl = jnp.float32(min(epoch / kl_warmup, 1.0) * kl_target)
    params, opt_state, loss, aux = train_step(params, opt_state, step_rng, lkl)
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | loss={aux['total']:.4f} | recon={aux['recon']:.4f} | "
              f"phys={aux['physics']:.4f} | smooth={aux['smooth']:.4f} | "
              f"small={aux['small']:.4f} | χ_max={aux['chi_max']:.4f}")

print("\n=== Evaluation ===")
mu, log_var = encoder.apply(params['encoder'], mag_grid_norm)
chi_pred = dec_susc.apply(params['dec_susc'], mu)
chi_pred_np = np.array(chi_pred)

all_coords_norm = jnp.array((cc - np.array(coord_mean)) / np.array(coord_std))

def predict_all(coord, chi_local):
    return dec_ilr.apply(params['dec_ilr'], mu, coord, chi_local)
ilr_pred_all = np.array(vmap(predict_all)(all_coords_norm, chi_pred))
comp_pred = ilr_inverse(ilr_pred_all)
Cu_pred = comp_pred[:, 0]

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res / ss_tot

chi_r2 = r2(susceptibility_true, chi_pred_np)
cu_r2 = r2(Cu, Cu_pred)
cu_corr = np.corrcoef(Cu, Cu_pred)[0, 1]

chi_val = chi_pred[val_cell_idx]
val_pred = np.array(vmap(predict_all)(val_coords_norm, chi_val))
val_rmse = float(np.sqrt(np.mean((val_pred - np.array(val_ilr))**2)))
chi_train = chi_pred[train_cell_idx]
tr_pred = np.array(vmap(predict_all)(train_coords_norm, chi_train))
tr_rmse = float(np.sqrt(np.mean((tr_pred - np.array(train_ilr))**2)))

print(f"χ pred: {chi_pred_np.min():.4f} – {chi_pred_np.max():.4f} (true: {susceptibility_true.min():.4f} – {susceptibility_true.max():.4f})")
print(f"Cu pred: {Cu_pred.min()*100:.2f}% – {Cu_pred.max()*100:.2f}% (true: {Cu.min()*100:.2f}% – {Cu.max()*100:.2f}%)")
print(f"\n=== RESULTS ===")
print(f"χ R²:           {chi_r2:.4f}")
print(f"Cu R² (global): {cu_r2:.4f}")
print(f"Cu correlation: {cu_corr:.4f}")
print(f"ILR train RMSE: {tr_rmse:.4f}")
print(f"ILR val RMSE:   {val_rmse:.4f}")

if cu_corr > 0.5 and chi_r2 > -1:
    print("\n✓ Model produces meaningful predictions")
else:
    print(f"\n✗ Still needs work")
