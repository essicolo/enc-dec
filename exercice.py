# %% [markdown]
# # Inversion géophysique avec encodeur-décodeur et PINN
#
# ## Le problème
#
# On a des mesures magnétiques en surface et quelques forages.
# On veut prédire les compositions (Cu, Ni, Co) partout en 3D.
#
# ## L'architecture
#
# ```
#                              ┌─────────────┐
#     Magnétique 2D ──────────▶│   ENCODER   │──────▶ Latent z
#                              └─────────────┘           │
#                                                        │
#                    ┌───────────────────────────────────┼───────────────────────┐
#                    │                                   │                       │
#                    ▼                                   ▼                       │
#           ┌────────────────┐                  ┌────────────────┐               │
#           │ DECODER_SUSC   │                  │  DECODER_ILR   │               │
#           │ latent → χ(3D) │                  │ latent+χ_local │               │
#           └────────────────┘                  │ +(x,y,z)→ ILR │               │
#                    │                          └────────────────┘               │
#                    │  ┌──────────────────────────────┘│                       │
#                    │  │ χ at borehole locations       │                       │
#                    ▼  ▼                               ▼                       │
#           ┌────────────────┐                   Loss reconstruction            │
#           │ FORWARD MODEL  │                   (vs forages)                   │
#           │  (physique)    │                                                  │
#           └────────────────┘                                                  │
#                    │                                                          │
#                    ▼                                                          │
#             Loss physique ◀───────────────────────────────────────────────────┘
#             (vs magnétique observé)
# ```
#
# Le PINN = la loss physique force la cohérence avec les observations.
# Le lien chi → ILR decoder force la cohérence physique-compositions.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
import optax

import lets_plot as lp
lp.LetsPlot.setup_html()

# %% [markdown]
# ---
# # Partie 1: Données synthétiques

# %%
from simpeg import maps
from simpeg.potential_fields import magnetics
from discretize import TensorMesh

# Mesh
nx, ny, nz = 16, 16, 8
dx, dy, dz = 250, 250, 125
mesh = TensorMesh(
    [np.ones(nx)*dx, np.ones(ny)*dy, np.ones(nz)*dz],
    origin=[0, 0, -nz*dz]
)
n_cells = mesh.nC
cc = mesh.cell_centers

print(f"Domaine: {nx*dx}m × {ny*dy}m × {nz*dz}m")
print(f"Cellules: {n_cells}")

# %%
# Deux corps minéralisés
d1 = np.sqrt(((cc[:,0]-1500)/400)**2 + ((cc[:,1]-1500)/350)**2 + ((cc[:,2]+500)/150)**2)
body1 = np.exp(-d1**2 * 2)

d2 = np.sqrt(((cc[:,0]-2800)/350)**2 + ((cc[:,1]-2500)/300)**2 + ((cc[:,2]+700)/120)**2)
body2 = np.exp(-d2**2 * 2)

# Susceptibilité
susceptibility_true = 0.001 + body1 * 0.05 + body2 * 0.02

# Compositions
Cu = 0.005 + body1 * 0.02 + body2 * 0.01
Ni = 0.003 + body1 * 0.01 + body2 * 0.025
Co = 0.001 + body1 * 0.005 + body2 * 0.012
gangue = 1 - Cu - Ni - Co
compositions = np.stack([Cu, Ni, Co, gangue], axis=1)

print(f"χ: {susceptibility_true.min():.4f} – {susceptibility_true.max():.4f}")
print(f"Cu: {Cu.min()*100:.2f}% – {Cu.max()*100:.2f}%")

# %% [markdown]
# ## ILR transform

# %%
# Matrice de contraste pour ILR (Helmert) — shared across forward/inverse
V_helmert = np.array([
    [np.sqrt(3/4), -1/np.sqrt(12), -1/np.sqrt(12), -1/np.sqrt(12)],
    [0, np.sqrt(2/3), -1/np.sqrt(6), -1/np.sqrt(6)],
    [0, 0, np.sqrt(1/2), -np.sqrt(1/2)]
])

def ilr_transform(comp):
    """Compositions (n, 4) → ILR (n, 3)"""
    log_comp = np.log(comp + 1e-10)
    return log_comp @ V_helmert.T

def ilr_inverse(ilr):
    """ILR (n, 3) → Compositions (n, 4)"""
    log_comp = ilr @ V_helmert
    comp = np.exp(log_comp)
    return comp / comp.sum(axis=1, keepdims=True)

ilr_true = ilr_transform(compositions)
print(f"ILR: {ilr_true.min():.2f} – {ilr_true.max():.2f}")

# %% [markdown]
# ## Forward magnétique avec SimPEG

# %%
# Stations
n_stations = 12
sx = np.linspace(300, nx*dx-300, n_stations)
sy = np.linspace(300, ny*dy-300, n_stations)
stations = np.array([[x, y, 50] for x in sx for y in sy])

# Survey
receivers = magnetics.receivers.Point(
    stations,        # OÙ on mesure
    components='tmi' # QU'EST-CE qu'on mesure (Total Magnetic Intensity)
)

# source: "quel champ magnétique ambiant?"
source = magnetics.sources.UniformBackgroundField(
    receiver_list=[receivers],
    amplitude=55000, # |B| en nT
    inclination=70, # angle vertical
    declination=0 # angle horizontal
)
survey = magnetics.Survey(source)

# Simulation — store_sensitivities='ram' to access the G matrix
sim = magnetics.Simulation3DIntegral(
    mesh=mesh,
    survey=survey,
    chiMap=maps.IdentityMap(mesh),
    active_cells=np.ones(n_cells, dtype=bool),
    store_sensitivities='ram'
)

# Reproducible noise with seeded numpy RNG
rng_np = np.random.default_rng(42)
mag_observed = sim.dpred(susceptibility_true) + rng_np.normal(0, 1.5, len(stations))

# Extract sensitivity matrix for exact JAX forward model
# The forward problem is linear: mag = G @ chi
G_matrix = sim.G
G_jax = jnp.array(G_matrix)

print(f"Magnétique: {mag_observed.min():.1f} – {mag_observed.max():.1f} nT")
print(f"Matrice de sensibilité G: {G_matrix.shape}")

# %%
# Visualiser
df_mag = pd.DataFrame({'x': stations[:,0], 'y': stations[:,1], 'mag': mag_observed})
(
    lp.ggplot(df_mag, lp.aes('x', 'y', fill='mag'))
    + lp.geom_tile()
    + lp.scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0)
    + lp.labs(title='Magnétique observé', x='X (m)', y='Y (m)', fill='nT')
)

# %% [markdown]
# ## Forages

# %%
# Hold out DH4 and DH7 for validation
# Add realistic assay noise (5% relative)

hole_xy = [
    (1500, 1500), (2800, 2500), (500, 500), (2000, 2000), (3500, 1000),
    (1200, 1800), (2500, 2200), (1800, 2800), (3000, 1500), (800, 2500)
]

validation_holes = {'DH4', 'DH7'}
noise_rel = 0.05  # 5% relative noise on assays

forages = []
for i, (hx, hy) in enumerate(hole_xy):
    ix, iy = np.argmin(np.abs(mesh.cell_centers_x - hx)), np.argmin(np.abs(mesh.cell_centers_y - hy))
    for iz in range(nz):
        idx = ix + iy * nx + iz * nx * ny
        # Add realistic assay noise
        cu_noisy = max(Cu[idx] * (1 + rng_np.normal(0, noise_rel)), 1e-6)
        ni_noisy = max(Ni[idx] * (1 + rng_np.normal(0, noise_rel)), 1e-6)
        co_noisy = max(Co[idx] * (1 + rng_np.normal(0, noise_rel)), 1e-6)
        gangue_noisy = max(1 - cu_noisy - ni_noisy - co_noisy, 1e-6)
        comp_noisy = np.array([[cu_noisy, ni_noisy, co_noisy, gangue_noisy]])
        ilr_noisy = ilr_transform(comp_noisy)[0]

        hole_name = f'DH{i+1}'
        forages.append({
            'hole': hole_name, 'x': mesh.cell_centers_x[ix], 'y': mesh.cell_centers_y[iy],
            'z': mesh.cell_centers_z[iz],
            'Cu': cu_noisy, 'Ni': ni_noisy, 'Co': co_noisy,
            'ilr0': ilr_noisy[0], 'ilr1': ilr_noisy[1], 'ilr2': ilr_noisy[2],
            'cell_idx': idx,
            'is_validation': hole_name in validation_holes
        })

df_forages = pd.DataFrame(forages)
df_train = df_forages[~df_forages['is_validation']].reset_index(drop=True)
df_val = df_forages[df_forages['is_validation']].reset_index(drop=True)

print(f"Échantillons total: {len(df_forages)} | train: {len(df_train)} | validation: {len(df_val)}")

# %% [markdown]
# ---
# # Partie 2: Forward model différentiable en JAX
#
# On utilise la matrice de sensibilité G de SimPEG plutôt qu'une
# approximation dipôle. Le problème forward étant linéaire (mag = G @ χ),
# c'est exact et directement différentiable en JAX.

# %%
@jit
def forward_magnetic_jax(susceptibility):
    """Forward magnétique exact via la matrice de sensibilité G de SimPEG."""
    return G_jax @ susceptibility

# Test
mag_test = forward_magnetic_jax(jnp.array(susceptibility_true))
print(f"Forward JAX (G matrix): {float(mag_test.min()):.1f} – {float(mag_test.max()):.1f}")
print(f"SimPEG dpred:           {mag_observed.min():.1f} – {mag_observed.max():.1f}")

# %% [markdown]
# ---
# # Partie 3: Architecture du réseau

# %%
# Normalization constants for susceptibility regularization
# Chi values are O(0.01), so squared losses are O(0.0001) without normalization.
# Dividing by CHI_NORM before computing MSE brings losses to O(1).
CHI_NORM = 0.01   # characteristic susceptibility scale
CHI_REF = 0.001   # background susceptibility for smallness penalty

# %%
# CNN encoder to preserve spatial structure of the magnetic grid
class Encoder(nn.Module):
    """Magnétique 2D → distribution sur le latent (CNN)"""
    latent_dim: int = 16

    @nn.compact
    def __call__(self, mag_grid):
        x = mag_grid[jnp.newaxis, :, :, jnp.newaxis]  # (1, 12, 12, 1)
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


# Wider decoder layers with softplus scaling
class DecoderSusceptibility(nn.Module):
    """Latent → susceptibilité 3D (une valeur par cellule)"""
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
        # softplus for numerically stable positive output
        return nn.softplus(raw) * 0.01


# KEY FIX: DecoderILR now receives chi_local — the predicted susceptibility
# at each location. This creates a direct information bridge from the
# physics-constrained susceptibility to composition predictions.
class DecoderILR(nn.Module):
    """Latent + chi_local + coordinates → ILR"""

    @nn.compact
    def __call__(self, latent, coord_norm, chi_local):
        # Normalize chi_local to O(1) so the network sees a meaningful feature
        chi_normalized = chi_local / CHI_NORM
        x = jnp.concatenate([latent, jnp.atleast_1d(chi_normalized), coord_norm])
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        return nn.Dense(3)(x)

# %% [markdown]
# ---
# # Partie 4: Loss function avec PINN

# %%
# Spatial smoothness regularization — normalized by CHI_NORM so loss is O(1)
def smoothness_loss(chi, nx, ny, nz):
    """Finite-difference smoothness on normalized susceptibility."""
    chi_n = chi / CHI_NORM
    chi_3d = chi_n.reshape(nz, ny, nx)
    diff_x = jnp.sum((chi_3d[:, :, 1:] - chi_3d[:, :, :-1])**2)
    diff_y = jnp.sum((chi_3d[:, 1:, :] - chi_3d[:, :-1, :])**2)
    diff_z = jnp.sum((chi_3d[1:, :, :] - chi_3d[:-1, :, :])**2)
    return (diff_x + diff_y + diff_z) / chi.size

def smallness_loss(chi):
    """Penalize deviation from background susceptibility (normalized)."""
    return jnp.mean(((chi - CHI_REF) / CHI_NORM)**2)


def loss_fn(params, encoder, dec_susc, dec_ilr,
            mag_grid, mag_observed,
            train_coords_norm, train_ilr, train_cell_idx,
            rng_key, lambda_physics=1.0, lambda_kl=0.01,
            lambda_smooth=0.1, lambda_small=0.05):
    """
    Loss totale = reconstruction ILR + KL + physique + smoothness + smallness

    Key: chi_pred is used both for the physics loss AND as input to the
    ILR decoder, creating a direct link between susceptibility and compositions.
    Regularization losses are normalized by CHI_NORM so they are O(1).
    """

    # --- Encoder ---
    mu, log_var = encoder.apply(params['encoder'], mag_grid)

    # Reparameterization trick
    std = jnp.exp(0.5 * log_var)
    eps = random.normal(rng_key, mu.shape)
    z = mu + std * eps

    # --- Decoder susceptibilité ---
    chi_pred = dec_susc.apply(params['dec_susc'], z)

    # --- Forward model (PINN) — exact G matrix ---
    mag_pred = forward_magnetic_jax(chi_pred)

    # Loss physique: normalized by data variance
    loss_physics = jnp.mean((mag_pred - mag_observed)**2) / jnp.var(mag_observed)

    # --- Regularization: properly scaled to O(1) ---
    loss_smooth = smoothness_loss(chi_pred, nx, ny, nz)
    loss_small = smallness_loss(chi_pred)

    # --- Decoder ILR — with chi_local input ---
    # Extract predicted chi at each training borehole location
    chi_at_train = chi_pred[train_cell_idx]

    def predict_ilr(coord, chi_local):
        return dec_ilr.apply(params['dec_ilr'], z, coord, chi_local)

    ilr_pred = vmap(predict_ilr)(train_coords_norm, chi_at_train)

    # Loss reconstruction ILR
    loss_recon = jnp.mean((ilr_pred - train_ilr)**2)

    # --- KL divergence ---
    loss_kl = -0.5 * jnp.mean(1 + log_var - mu**2 - jnp.exp(log_var))

    # --- Total ---
    loss_total = (loss_recon
                  + lambda_kl * loss_kl
                  + lambda_physics * loss_physics
                  + lambda_smooth * loss_smooth
                  + lambda_small * loss_small)

    return loss_total, {
        'total': loss_total, 'recon': loss_recon,
        'kl': loss_kl, 'physics': loss_physics,
        'smooth': loss_smooth, 'small': loss_small,
        'chi_mean': jnp.mean(chi_pred), 'chi_max': jnp.max(chi_pred)
    }

# %% [markdown]
# ---
# # Partie 5: Entraînement

# %%
# Initialiser
rng = random.PRNGKey(42)
rng, *init_rngs = random.split(rng, 4)

latent_dim = 16  # reduced from 32 to avoid overfitting with 144 measurements
encoder = Encoder(latent_dim=latent_dim)
dec_susc = DecoderSusceptibility(n_cells=n_cells)
dec_ilr = DecoderILR()

# Dummy inputs
dummy_mag = jnp.zeros((n_stations, n_stations))
dummy_latent = jnp.zeros(latent_dim)
dummy_coord = jnp.zeros(3)
dummy_chi = jnp.float32(0.0)

params = {
    'encoder': encoder.init(init_rngs[0], dummy_mag),
    'dec_susc': dec_susc.init(init_rngs[1], dummy_latent),
    'dec_ilr': dec_ilr.init(init_rngs[2], dummy_latent, dummy_coord, dummy_chi)
}

# %%
# Préparer les données — use training set only
mag_grid = jnp.array(mag_observed.reshape(n_stations, n_stations))
mag_grid_norm = (mag_grid - mag_grid.mean()) / (mag_grid.std() + 1e-6)

mag_obs_jax = jnp.array(mag_observed)

# Coordonnées des forages normalisées (training only)
coord_mean = jnp.array([nx*dx/2, ny*dy/2, -nz*dz/2])
coord_std = jnp.array([nx*dx/2, ny*dy/2, nz*dz/2])
train_coords_norm = jnp.array((df_train[['x', 'y', 'z']].values - np.array(coord_mean)) / np.array(coord_std))
train_ilr = jnp.array(df_train[['ilr0', 'ilr1', 'ilr2']].values)
train_cell_idx = jnp.array(df_train['cell_idx'].values, dtype=jnp.int32)

# Validation coordinates
val_coords_norm = jnp.array((df_val[['x', 'y', 'z']].values - np.array(coord_mean)) / np.array(coord_std))
val_ilr = jnp.array(df_val[['ilr0', 'ilr1', 'ilr2']].values)
val_cell_idx = jnp.array(df_val['cell_idx'].values, dtype=jnp.int32)

print(f"Prêt pour l'entraînement")

# %%
# Learning rate schedule with warmup cosine decay
n_epochs = 5000
schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4,
    peak_value=1e-3,
    warmup_steps=200,
    decay_steps=n_epochs
)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(params)

# KL annealing parameters
kl_warmup_epochs = 1000
kl_target = 0.01

@jit
def train_step(params, opt_state, rng_key, lambda_kl):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, encoder, dec_susc, dec_ilr,
        mag_grid_norm, mag_obs_jax,
        train_coords_norm, train_ilr, train_cell_idx,
        rng_key, lambda_physics=1.0, lambda_kl=lambda_kl,
        lambda_smooth=0.1,   # properly scaled — chi normalized to O(1)
        lambda_small=0.05    # penalize deviation from background
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# %%
# Boucle d'entraînement avec KL annealing
history = []

for epoch in range(n_epochs):
    rng, step_rng = random.split(rng)

    # KL annealing — linearly increase from 0 to kl_target
    lambda_kl = jnp.float32(min(epoch / kl_warmup_epochs, 1.0) * kl_target)

    params, opt_state, loss, aux = train_step(params, opt_state, step_rng, lambda_kl)
    history.append({**{k: float(v) for k, v in aux.items()}, 'epoch': epoch})

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | loss={aux['total']:.4f} | recon={aux['recon']:.4f} | "
              f"phys={aux['physics']:.4f} | smooth={aux['smooth']:.4f} | "
              f"small={aux['small']:.4f} | χ_max={aux['chi_max']:.4f}")

# %%
# Courbes de loss
df_hist = pd.DataFrame(history)

(
    lp.ggplot(df_hist, lp.aes(x='epoch'))
    + lp.geom_line(lp.aes(y='recon'), color="#962561")
    + lp.geom_line(lp.aes(y='physics'), color="#abc000")
    + lp.geom_line(lp.aes(y='smooth'), color="#00abc0")
    + lp.geom_line(lp.aes(y='small'), color="#c06000")
    + lp.scale_y_log10()
    + lp.labs(title='Entraînement', x='Epoch', y='Loss (log)')
)

# %% [markdown]
# ---
# # Partie 6: Prédictions

# %%
# Encoder → latent
mu, log_var = encoder.apply(params['encoder'], mag_grid_norm)

# Décoder la susceptibilité
chi_pred = dec_susc.apply(params['dec_susc'], mu)
chi_pred_np = np.array(chi_pred)

print(f"χ prédit: {chi_pred_np.min():.4f} – {chi_pred_np.max():.4f}")
print(f"χ vrai:   {susceptibility_true.min():.4f} – {susceptibility_true.max():.4f}")

# %%
# Vérifier le forward
mag_from_pred = forward_magnetic_jax(chi_pred)
print(f"Mag prédit: {float(mag_from_pred.min()):.1f} – {float(mag_from_pred.max()):.1f}")
print(f"Mag observé: {mag_observed.min():.1f} – {mag_observed.max():.1f}")

# %% [markdown]
# ## Prédire les ILR avec incertitude (sampling)

# %%
# Grille de prédiction: toutes les cellules
all_coords_norm = (cc - np.array(coord_mean)) / np.array(coord_std)
all_coords_norm_jax = jnp.array(all_coords_norm)

# Sample from latent posterior for uncertainty quantification
n_samples = 100
ilr_samples = []

for i in range(n_samples):
    rng, sample_rng = random.split(rng)
    std = jnp.exp(0.5 * log_var)
    eps = random.normal(sample_rng, mu.shape)
    z = mu + std * eps

    # Recompute chi for this latent sample (uncertainty propagates through chi)
    chi_sample = dec_susc.apply(params['dec_susc'], z)

    def predict_ilr(coord, chi_local):
        return dec_ilr.apply(params['dec_ilr'], z, coord, chi_local)

    ilr_sample = vmap(predict_ilr)(all_coords_norm_jax, chi_sample)
    ilr_samples.append(np.array(ilr_sample))

ilr_samples = np.stack(ilr_samples)
ilr_mean = ilr_samples.mean(axis=0)
ilr_std = ilr_samples.std(axis=0)

print(f"ILR prédit: {ilr_mean.min():.2f} – {ilr_mean.max():.2f}")
print(f"Incertitude: {ilr_std.min():.3f} – {ilr_std.max():.3f}")

# %%
# Convertir en compositions
comp_pred = ilr_inverse(ilr_mean)
Cu_pred = comp_pred[:, 0]

print(f"Cu prédit: {Cu_pred.min()*100:.2f}% – {Cu_pred.max()*100:.2f}%")
print(f"Cu vrai:   {Cu.min()*100:.2f}% – {Cu.max()*100:.2f}%")

# %% [markdown]
# ## Métriques de validation

# %%
# Predict ILR at borehole locations using the mean latent
def predict_ilr_at_coords(mu_latent, coords_norm, cell_idx):
    chi_val = dec_susc.apply(params['dec_susc'], mu_latent)
    chi_at_coords = chi_val[cell_idx]
    def predict(coord, chi_local):
        return dec_ilr.apply(params['dec_ilr'], mu_latent, coord, chi_local)
    return vmap(predict)(coords_norm, chi_at_coords)

# Training metrics
train_ilr_pred = predict_ilr_at_coords(mu, train_coords_norm, train_cell_idx)
train_rmse = float(jnp.sqrt(jnp.mean((train_ilr_pred - train_ilr)**2)))
ss_res_train = float(jnp.sum((train_ilr_pred - train_ilr)**2))
ss_tot_train = float(jnp.sum((train_ilr - jnp.mean(train_ilr, axis=0))**2))
train_r2 = 1 - ss_res_train / ss_tot_train

# Validation metrics
val_ilr_pred = predict_ilr_at_coords(mu, val_coords_norm, val_cell_idx)
val_rmse = float(jnp.sqrt(jnp.mean((val_ilr_pred - val_ilr)**2)))
ss_res_val = float(jnp.sum((val_ilr_pred - val_ilr)**2))
ss_tot_val = float(jnp.sum((val_ilr - jnp.mean(val_ilr, axis=0))**2))
val_r2 = 1 - ss_res_val / ss_tot_val

# Susceptibility metrics
chi_rmse = float(np.sqrt(np.mean((chi_pred_np - susceptibility_true)**2)))
ss_res_chi = float(np.sum((chi_pred_np - susceptibility_true)**2))
ss_tot_chi = float(np.sum((susceptibility_true - susceptibility_true.mean())**2))
chi_r2 = 1 - ss_res_chi / ss_tot_chi

print(f"--- Métriques de validation ---")
print(f"ILR Train  RMSE: {train_rmse:.4f} | R²: {train_r2:.4f}")
print(f"ILR Val    RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
print(f"χ global   RMSE: {chi_rmse:.6f} | R²: {chi_r2:.4f}")

# %% [markdown]
# ---
# # Partie 7: Visualisations

# %%
# Coupe horizontale à mi-profondeur
iz = nz // 2
idx_slice = slice(iz * nx * ny, (iz + 1) * nx * ny)

df_results = pd.DataFrame({
    'x': cc[idx_slice, 0],
    'y': cc[idx_slice, 1],
    'chi_pred': chi_pred_np[idx_slice],
    'chi_true': susceptibility_true[idx_slice],
    'Cu_pred': Cu_pred[idx_slice] * 100,
    'Cu_true': Cu[idx_slice] * 100,
    'uncertainty': ilr_std[idx_slice, 0]
})

# Distinguish train vs validation boreholes on plots
df_holes_train = df_train.drop_duplicates('hole')[['x', 'y']]
df_holes_val = df_val.drop_duplicates('hole')[['x', 'y']]

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping= lp.aes('x', 'y', fill='chi_pred'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkblue', name='χ (SI)')
    + lp.labs(title=f'Susceptibilité prédite (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('x', 'y'))
    + lp.geom_tile(mapping=lp.aes(fill='chi_true'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkblue', name='χ (SI)')
    + lp.labs(title=f'Susceptibilité vraie (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('x', 'y'))
    + lp.geom_tile(mapping=lp.aes(fill='Cu_pred'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkred', name='Cu %')
    + lp.labs(title=f'Cu prédit (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('x', 'y', fill='Cu_true'))
    + lp.geom_tile()
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkred', name='Cu %')
    + lp.labs(title=f'Cu vrai (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('x', 'y'))
    + lp.geom_tile(mapping=lp.aes(fill='uncertainty'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='purple', name='σ')
    + lp.labs(title=f'Incertitude sur ILR[0] (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %% [markdown]
# ## Comparaison quantitative

# %%
(
    lp.ggplot(df_results, lp.aes('chi_true', 'chi_pred'))
    + lp.geom_point(alpha=0.5)
    + lp.geom_abline(slope=1, intercept=0, color='red', linetype='dashed')
    + lp.labs(title='Susceptibilité: prédit vs vrai', x='χ vrai', y='χ prédit')
)

# %%
(
    lp.ggplot(df_results, lp.aes('Cu_true', 'Cu_pred'))
    + lp.geom_point(alpha=0.5)
    + lp.geom_abline(slope=1, intercept=0, color='red', linetype='dashed')
    + lp.labs(title='Cu: prédit vs vrai', x='Cu vrai (%)', y='Cu prédit (%)')
)

# %% [markdown]
# ---
# # Résumé
#
# ## Ce qu'on a fait
#
# 1. **SimPEG** pour générer des données magnétiques réalistes
# 2. **ILR** pour transformer les compositions (fermeture respectée)
# 3. **Encodeur-décodeur avec PINN**:
#    - Encoder CNN: magnétique 2D → latent (préserve la structure spatiale)
#    - Decoder susceptibilité: latent → χ 3D (softplus, bien dimensionné)
#    - Decoder ILR: latent + χ_local/CHI_NORM + (x,y,z) → compositions
#    - Forward exact: χ → magnétique prédit via matrice G de SimPEG
#    - Loss physique: ||mag_prédit - mag_observé||² / var(observé)
#    - Régularisation: lissage + smallness, normalisés par CHI_NORM
# 4. **Incertitude** par sampling du latent (100 échantillons)
# 5. **Validation** sur forages retenus (DH4, DH7) avec RMSE et R²
#
# ## Améliorations clés
#
# - χ_local normalisé comme entrée au décodeur ILR: lien direct physique → compositions
# - Matrice G de SimPEG pour forward exact (au lieu d'approximation dipôle)
# - Régularisation normalisée par CHI_NORM: smoothness + smallness à O(1)
# - CNN encoder, softplus, KL annealing, LR schedule cosine
# - Bruit réaliste sur forages, validation train/test
