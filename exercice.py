# %% [markdown]
# # Multi-Sensor Geophysical Inversion with Encoder-Decoder PINN
#
# ## Exploration Scenario
#
# A Cu-Ni-Co sulfide exploration program over a 4km × 4km area.
# Multiple data sources are integrated through a shared latent space
# for joint inversion — inspired by KoBold Metals' multi-sensor approach.
#
# ### Data Sources
#
# | Sensor              | Type            | Coverage          | Physics             |
# |---------------------|-----------------|-------------------|---------------------|
# | Airborne magnetics  | Geophysical     | 20×20 grid, z=80m | TMI → susceptibility|
# | Ground gravity      | Geophysical     | 20×20 grid, z=1m  | gz → density        |
# | Hyperspectral drone | Remote sensing  | 16×16×10 bands    | Reflectance spectra |
# | Soil geochemistry   | Geochemical     | 50 samples → grid | Cu,Ni,Co,Fe,S       |
# | Drill holes         | Direct sampling | 10 holes, 8 each  | Cu,Ni,Co assays     |
#
# ### Architecture
#
# ```
#                ┌──────────────┐
# Magnetics ────▶│ CNN Branch 1 │──┐
#                └──────────────┘  │
#                ┌──────────────┐  │    ┌─────────┐
# Gravity ──────▶│ CNN Branch 2 │──┼───▶│ FUSION  │──▶ Latent z
#                └──────────────┘  │    │  MLP    │       │
#                ┌──────────────┐  │    └─────────┘       │
# Hyperspectral ▶│ CNN Branch 3 │──┤                      │
#  (10 bands)    └──────────────┘  │    For each cell (x,y,z):
#                ┌──────────────┐  │    ┌───────────────────────────────┐
# Geochemistry ─▶│ CNN Branch 4 │──┘    │         (z, coord)            │
#  (gridded)     └──────────────┘       │  ┌──────────┐ ┌──────────┐   │
#                                       │  │DEC_SUSC  │ │DEC_DENS  │   │
#                                       │  │→ χ_local │ │→ ρ_local │   │
#                                       │  └────┬─────┘ └────┬─────┘   │
#                                       │       ▼            ▼         │
#                                       │  ┌──────────────────────┐    │
#                                       │  │     DECODER_ILR      │    │
#                                       │  │ z+coord+χ+ρ → ILR   │    │
#                                       │  └──────────┬───────────┘    │
#                                       └─────────────┼────────────────┘
#                            ┌────────────┬───────────┘
#                            ▼            ▼
#                     ┌────────────┐ ┌────────────┐  ┌──────────────┐
#                     │ FORWARD MAG│ │FORWARD GRAV│  │Loss reconstr.│
#                     │  G_mag @ χ │ │ G_grav @ Δρ│  │(vs boreholes)│
#                     └─────┬──────┘ └─────┬──────┘  └──────────────┘
#                           ▼              ▼
#                     Loss magnetics  Loss gravity
# ```
#
# **Key design choices**:
# 1. Surface sensors are encoder inputs only — no decoder losses on them
#    (avoids parasitic autoencoder shortcuts).
# 2. Coordinate-conditioned decoders (NeRF-style): each cell gets (z, x, y, z)
#    → chi/rho, with shared weights providing implicit spatial regularization.
# 3. Deterministic encoder (no VAE sampling) — KL loss fights recon loss
#    in single-scene problems.

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
from scipy.interpolate import griddata

import lets_plot as lp
lp.LetsPlot.setup_html()

# %% [markdown]
# ---
# # Part 1: Synthetic Ground Truth

# %%
from simpeg import maps
from simpeg.potential_fields import magnetics, gravity
from discretize import TensorMesh

# 3D mesh: 4km × 4km × 1km deep
nx, ny, nz = 16, 16, 8
dx, dy, dz = 250, 250, 125
mesh = TensorMesh(
    [np.ones(nx)*dx, np.ones(ny)*dy, np.ones(nz)*dz],
    origin=[0, 0, -nz*dz]
)
n_cells = mesh.nC
cc = mesh.cell_centers

print(f"Domain: {nx*dx}m × {ny*dy}m × {nz*dz}m")
print(f"Cells: {n_cells}")

# %%
# Two mineralized bodies: Cu-Ni sulfide lenses
d1 = np.sqrt(((cc[:,0]-1500)/400)**2 + ((cc[:,1]-1500)/350)**2 + ((cc[:,2]+500)/150)**2)
body1 = np.exp(-d1**2 * 2)

d2 = np.sqrt(((cc[:,0]-2800)/350)**2 + ((cc[:,1]-2500)/300)**2 + ((cc[:,2]+700)/120)**2)
body2 = np.exp(-d2**2 * 2)

# Magnetic susceptibility (SI) — sulfides are moderately magnetic
susceptibility_true = 0.001 + body1 * 0.05 + body2 * 0.02

# Density contrast (g/cc) — massive sulfides are denser than host rock
density_true = 0.0 + body1 * 0.5 + body2 * 0.3

# Mineral compositions (mass fractions)
Cu = 0.005 + body1 * 0.02 + body2 * 0.01
Ni = 0.003 + body1 * 0.01 + body2 * 0.025
Co = 0.001 + body1 * 0.005 + body2 * 0.012
gangue = 1 - Cu - Ni - Co
compositions = np.stack([Cu, Ni, Co, gangue], axis=1)

print(f"χ:  {susceptibility_true.min():.4f} – {susceptibility_true.max():.4f} SI")
print(f"Δρ: {density_true.min():.2f} – {density_true.max():.2f} g/cc")
print(f"Cu: {Cu.min()*100:.2f}% – {Cu.max()*100:.2f}%")

# %% [markdown]
# ## ILR Transform

# %%
# Helmert contrast matrix for ILR (isometric log-ratio)
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
print(f"ILR range: {ilr_true.min():.2f} – {ilr_true.max():.2f}")

# %% [markdown]
# ---
# # Part 2: Multi-Sensor Surveys

# %% [markdown]
# ## 2a. Airborne Magnetic Survey (TMI)
#
# Helicopter-borne magnetometer at 80m flight height.
# 400 stations on a regular 20×20 grid.

# %%
rng_np = np.random.default_rng(42)

n_stations_mag = 20
sx_mag = np.linspace(200, nx*dx-200, n_stations_mag)
sy_mag = np.linspace(200, ny*dy-200, n_stations_mag)
stations_mag = np.array([[x, y, 80.0] for x in sx_mag for y in sy_mag])

receivers_mag = magnetics.receivers.Point(stations_mag, components='tmi')
source_mag = magnetics.sources.UniformBackgroundField(
    receiver_list=[receivers_mag],
    amplitude=55000, inclination=70, declination=0
)
survey_mag = magnetics.Survey(source_mag)

sim_mag = magnetics.Simulation3DIntegral(
    mesh=mesh, survey=survey_mag,
    chiMap=maps.IdentityMap(mesh),
    active_cells=np.ones(n_cells, dtype=bool),
    store_sensitivities='ram'
)

mag_observed = sim_mag.dpred(susceptibility_true) + rng_np.normal(0, 1.5, len(stations_mag))
G_mag = sim_mag.G
G_mag_jax = jnp.array(G_mag)

print(f"Magnetic TMI: {mag_observed.min():.1f} – {mag_observed.max():.1f} nT")
print(f"G_mag shape: {G_mag.shape}")

# %%
df_mag = pd.DataFrame({'x': stations_mag[:,0], 'y': stations_mag[:,1], 'mag': mag_observed})
(
    lp.ggplot(df_mag, lp.aes('x', 'y', fill='mag'))
    + lp.geom_tile()
    + lp.scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0)
    + lp.labs(title='Airborne Magnetics (TMI)', x='X (m)', y='Y (m)', fill='nT')
)

# %% [markdown]
# ## 2b. Ground Gravity Survey (gz)
#
# Ground-based gravimeter measuring vertical gravity anomaly.
# Same horizontal grid, different elevation (z=1m).

# %%
stations_grav = np.array([[x, y, 1.0] for x in sx_mag for y in sy_mag])

receivers_grav = gravity.receivers.Point(stations_grav, components='gz')
source_grav = gravity.sources.SourceField(receiver_list=[receivers_grav])
survey_grav = gravity.survey.Survey(source_field=source_grav)

sim_grav = gravity.Simulation3DIntegral(
    mesh=mesh, survey=survey_grav,
    rhoMap=maps.IdentityMap(mesh),
    active_cells=np.ones(n_cells, dtype=bool),
    store_sensitivities='ram'
)

grav_observed = sim_grav.dpred(density_true) + rng_np.normal(0, 0.02, len(stations_grav))
G_grav = sim_grav.G
G_grav_jax = jnp.array(G_grav)

print(f"Gravity gz: {grav_observed.min():.3f} – {grav_observed.max():.3f} mGal")
print(f"G_grav shape: {G_grav.shape}")

# %%
df_grav = pd.DataFrame({'x': stations_grav[:,0], 'y': stations_grav[:,1], 'gz': grav_observed})
(
    lp.ggplot(df_grav, lp.aes('x', 'y', fill='gz'))
    + lp.geom_tile()
    + lp.scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0)
    + lp.labs(title='Ground Gravity (gz)', x='X (m)', y='Y (m)', fill='mGal')
)

# %% [markdown]
# ## 2c. Hyperspectral Drone Survey
#
# Drone-mounted hyperspectral sensor capturing 10 spectral bands
# from VNIR (0.4 μm) to SWIR (2.5 μm). Alteration minerals around
# sulfide deposits create diagnostic absorption features.

# %%
# Surface anomaly model: mineralization halos projected to surface
# In reality, weathering and supergene enrichment create broader,
# attenuated surface expressions of deeper deposits.
surface_xx, surface_yy = np.meshgrid(mesh.cell_centers_x, mesh.cell_centers_y)

d1_surf = np.sqrt(((surface_xx - 1500)/600)**2 + ((surface_yy - 1500)/550)**2)
body1_surf = np.exp(-d1_surf**2 * 1.5) * 0.4

d2_surf = np.sqrt(((surface_xx - 2800)/550)**2 + ((surface_yy - 2500)/500)**2)
body2_surf = np.exp(-d2_surf**2 * 1.5) * 0.25

# Spectral endmembers: reflectance profiles for different surface materials
# Bands span VNIR to SWIR (iron oxides, clays, carbonates, silica)
n_bands = 10
endmember_bg = np.array([0.30, 0.32, 0.35, 0.33, 0.30, 0.28, 0.25, 0.27, 0.30, 0.28])
endmember_fe = np.array([0.15, 0.22, 0.38, 0.55, 0.68, 0.62, 0.48, 0.42, 0.38, 0.33])  # gossan (iron oxide)
endmember_clay = np.array([0.40, 0.45, 0.50, 0.52, 0.58, 0.50, 0.40, 0.28, 0.18, 0.22]) # clay alteration

# Linear spectral mixing at surface
frac_fe = body1_surf + 0.5 * body2_surf     # iron oxide fraction
frac_clay = 0.5 * body1_surf + body2_surf   # clay fraction
frac_bg = np.clip(1.0 - frac_fe - frac_clay, 0.1, 1.0)
total = frac_bg + frac_fe + frac_clay
frac_bg /= total; frac_fe /= total; frac_clay /= total

hyper_image = (frac_bg[:,:,None] * endmember_bg +
               frac_fe[:,:,None] * endmember_fe +
               frac_clay[:,:,None] * endmember_clay)
hyper_image += rng_np.normal(0, 0.02, hyper_image.shape)
hyper_image = np.clip(hyper_image, 0, 1)

print(f"Hyperspectral image: {hyper_image.shape} (ny × nx × bands)")
print(f"Reflectance range: {hyper_image.min():.3f} – {hyper_image.max():.3f}")

# %%
# Show RGB composite (bands 3, 2, 1) and SWIR (bands 8, 6, 4)
df_hyper_rgb = pd.DataFrame({
    'x': surface_xx.ravel(), 'y': surface_yy.ravel(),
    'band_3': hyper_image[:,:,2].ravel()
})
(
    lp.ggplot(df_hyper_rgb, lp.aes('x', 'y', fill='band_3'))
    + lp.geom_tile()
    + lp.scale_fill_gradient(low='#1a1a2e', high='#e94560', name='Refl.')
    + lp.labs(title='Hyperspectral Band 3 (VNIR red)', x='X (m)', y='Y (m)')
)

# %% [markdown]
# ## 2d. Surface Geochemistry (Soil Samples)
#
# 50 soil samples analyzed for Cu, Ni, Co, Fe, S (ppm or %).
# Interpolated to a 16×16 grid for the encoder.

# %%
n_soil = 50
soil_x = rng_np.uniform(200, nx*dx-200, n_soil)
soil_y = rng_np.uniform(200, ny*dy-200, n_soil)

# Evaluate surface body intensities at sample locations
d1_soil = np.sqrt(((soil_x - 1500)/600)**2 + ((soil_y - 1500)/550)**2)
body1_soil = np.exp(-d1_soil**2 * 1.5) * 0.4
d2_soil = np.sqrt(((soil_x - 2800)/550)**2 + ((soil_y - 2500)/500)**2)
body2_soil = np.exp(-d2_soil**2 * 1.5) * 0.25

# Realistic soil geochemistry concentrations with noise
Cu_soil = 30 + body1_soil * 500 + body2_soil * 300 + rng_np.normal(0, 10, n_soil)  # ppm
Ni_soil = 20 + body1_soil * 200 + body2_soil * 600 + rng_np.normal(0, 8, n_soil)   # ppm
Co_soil = 8 + body1_soil * 80 + body2_soil * 200 + rng_np.normal(0, 4, n_soil)     # ppm
Fe_soil = 3.0 + body1_soil * 5 + body2_soil * 3 + rng_np.normal(0, 0.3, n_soil)    # %
S_soil = 0.1 + body1_soil * 2 + body2_soil * 1.5 + rng_np.normal(0, 0.05, n_soil)  # %
Cu_soil = np.clip(Cu_soil, 5, None)
Ni_soil = np.clip(Ni_soil, 5, None)
Co_soil = np.clip(Co_soil, 2, None)

df_soil = pd.DataFrame({
    'x': soil_x, 'y': soil_y,
    'Cu_ppm': Cu_soil, 'Ni_ppm': Ni_soil, 'Co_ppm': Co_soil,
    'Fe_pct': Fe_soil, 'S_pct': S_soil
})

# Interpolate to 16×16 grid for encoder input
grid_xy = np.column_stack([surface_xx.ravel(), surface_yy.ravel()])
soil_points = np.column_stack([soil_x, soil_y])

geochem_grid = np.zeros((nx * ny, 5))
for i, col in enumerate(['Cu_ppm', 'Ni_ppm', 'Co_ppm', 'Fe_pct', 'S_pct']):
    vals = df_soil[col].values
    grid_lin = griddata(soil_points, vals, grid_xy, method='linear')
    grid_near = griddata(soil_points, vals, grid_xy, method='nearest')
    geochem_grid[:, i] = np.where(np.isnan(grid_lin), grid_near, grid_lin)

# Normalize each channel to [0, 1]
geochem_min = geochem_grid.min(axis=0)
geochem_max = geochem_grid.max(axis=0)
geochem_grid_norm = (geochem_grid - geochem_min) / (geochem_max - geochem_min + 1e-8)
geochem_grid_2d = geochem_grid_norm.reshape(ny, nx, 5)

print(f"Soil samples: {n_soil}")
print(f"Geochem grid: {geochem_grid_2d.shape} (ny × nx × elements)")

# %%
(
    lp.ggplot(df_soil, lp.aes('x', 'y', color='Cu_ppm'))
    + lp.geom_point(size=4)
    + lp.scale_color_gradient(low='lightyellow', high='darkred', name='Cu (ppm)')
    + lp.labs(title='Soil Geochemistry — Cu', x='X (m)', y='Y (m)')
)

# %% [markdown]
# ## 2e. Drill Holes
#
# 10 diamond drill holes with Cu, Ni, Co assays at 8 depth intervals.
# DH4 and DH7 are held out for validation.

# %%
hole_xy = [
    (1500, 1500), (2800, 2500), (500, 500), (2000, 2000), (3500, 1000),
    (1200, 1800), (2500, 2200), (1800, 2800), (3000, 1500), (800, 2500)
]

validation_holes = {'DH4', 'DH7'}
noise_rel = 0.05  # 5% relative noise on assays

forages = []
for i, (hx, hy) in enumerate(hole_xy):
    ix = np.argmin(np.abs(mesh.cell_centers_x - hx))
    iy = np.argmin(np.abs(mesh.cell_centers_y - hy))
    for iz in range(nz):
        idx = ix + iy * nx + iz * nx * ny
        cu_noisy = max(Cu[idx] * (1 + rng_np.normal(0, noise_rel)), 1e-6)
        ni_noisy = max(Ni[idx] * (1 + rng_np.normal(0, noise_rel)), 1e-6)
        co_noisy = max(Co[idx] * (1 + rng_np.normal(0, noise_rel)), 1e-6)
        gangue_noisy = max(1 - cu_noisy - ni_noisy - co_noisy, 1e-6)
        comp_noisy = np.array([[cu_noisy, ni_noisy, co_noisy, gangue_noisy]])
        ilr_noisy = ilr_transform(comp_noisy)[0]

        hole_name = f'DH{i+1}'
        forages.append({
            'hole': hole_name,
            'x': mesh.cell_centers_x[ix], 'y': mesh.cell_centers_y[iy],
            'z': mesh.cell_centers_z[iz],
            'Cu': cu_noisy, 'Ni': ni_noisy, 'Co': co_noisy,
            'ilr0': ilr_noisy[0], 'ilr1': ilr_noisy[1], 'ilr2': ilr_noisy[2],
            'cell_idx': idx,
            'is_validation': hole_name in validation_holes
        })

df_forages = pd.DataFrame(forages)
df_train = df_forages[~df_forages['is_validation']].reset_index(drop=True)
df_val = df_forages[df_forages['is_validation']].reset_index(drop=True)

print(f"Drill samples: {len(df_forages)} | train: {len(df_train)} | val: {len(df_val)}")

# %% [markdown]
# ---
# # Part 3: Differentiable Forward Models (JAX)
#
# Both forward problems are linear, so we use the sensitivity matrices
# extracted from SimPEG for exact, differentiable forward modeling.

# %%
@jit
def forward_magnetic_jax(susceptibility):
    """Forward magnetics: TMI = G_mag @ χ"""
    return G_mag_jax @ susceptibility

@jit
def forward_gravity_jax(density):
    """Forward gravity: gz = G_grav @ Δρ"""
    return G_grav_jax @ density

# Verify forward models
mag_test = forward_magnetic_jax(jnp.array(susceptibility_true))
grav_test = forward_gravity_jax(jnp.array(density_true))
print(f"Forward magnetics: {float(mag_test.min()):.1f} – {float(mag_test.max()):.1f} nT")
print(f"Forward gravity:   {float(grav_test.min()):.4f} – {float(grav_test.max()):.4f} mGal")

# %% [markdown]
# ---
# # Part 4: Network Architecture
#
# Multi-modal encoder with separate branches for each sensor,
# fused through a shared MLP into a single latent space.
# Two physical property decoders + one composition decoder.

# %%
# No explicit normalization constants needed — the coordinate-conditioned
# decoders use softplus * scale directly, and chi/rho enter the ILR decoder
# as chi*100 and rho*10 for numerical balance.

# %%
class MultiModalEncoder(nn.Module):
    """
    Fuses magnetic, gravity, hyperspectral, and geochemistry inputs
    into a shared latent distribution (VAE-style).
    """
    latent_dim: int = 32

    @nn.compact
    def __call__(self, mag_grid, grav_grid, hyper_img, geochem_grid):
        # --- Magnetic branch: (20, 20) → 64 features ---
        m = mag_grid[jnp.newaxis, :, :, jnp.newaxis]
        m = nn.Conv(16, kernel_size=(3, 3), padding='SAME')(m)
        m = nn.relu(m)
        m = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(m)
        m = nn.relu(m)
        m = m.ravel()
        m = nn.Dense(64)(m)
        m = nn.relu(m)

        # --- Gravity branch: (20, 20) → 64 features ---
        g = grav_grid[jnp.newaxis, :, :, jnp.newaxis]
        g = nn.Conv(16, kernel_size=(3, 3), padding='SAME')(g)
        g = nn.relu(g)
        g = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(g)
        g = nn.relu(g)
        g = g.ravel()
        g = nn.Dense(64)(g)
        g = nn.relu(g)

        # --- Hyperspectral branch: (16, 16, 10) → 64 features ---
        h = hyper_img[jnp.newaxis, :, :, :]
        h = nn.Conv(16, kernel_size=(3, 3), padding='SAME')(h)
        h = nn.relu(h)
        h = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(h)
        h = nn.relu(h)
        h = h.ravel()
        h = nn.Dense(64)(h)
        h = nn.relu(h)

        # --- Geochemistry branch: (16, 16, 5) → 32 features ---
        c = geochem_grid[jnp.newaxis, :, :, :]
        c = nn.Conv(16, kernel_size=(3, 3), padding='SAME')(c)
        c = nn.relu(c)
        c = c.ravel()
        c = nn.Dense(32)(c)
        c = nn.relu(c)

        # --- Fusion MLP ---
        fused = jnp.concatenate([m, g, h, c])  # 64+64+64+32 = 224
        fused = nn.Dense(128)(fused)
        fused = nn.relu(fused)
        fused = nn.Dense(64)(fused)
        fused = nn.relu(fused)

        mu = nn.Dense(self.latent_dim)(fused)
        log_var = nn.Dense(self.latent_dim)(fused)
        return mu, log_var


class DecoderSusceptibility(nn.Module):
    """
    (latent, coord_norm) → χ_local (scalar).
    Coordinate-conditioned decoder (NeRF-style): each cell gets
    a position-dependent prediction from the shared latent vector.
    Weight sharing across all cells provides implicit spatial regularization.
    """

    @nn.compact
    def __call__(self, latent, coord_norm):
        x = jnp.concatenate([latent, coord_norm])
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        return nn.softplus(nn.Dense(1)(x).squeeze()) * 0.01


class DecoderDensity(nn.Module):
    """
    (latent, coord_norm) → Δρ_local (scalar).
    Same coordinate-conditioned architecture as susceptibility decoder.
    """

    @nn.compact
    def __call__(self, latent, coord_norm):
        x = jnp.concatenate([latent, coord_norm])
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        return nn.softplus(nn.Dense(1)(x).squeeze()) * 0.1


class DecoderILR(nn.Module):
    """
    Latent + local physical properties + coordinates → ILR compositions.
    Both chi_local and rho_local provide direct physical bridges from
    the geophysical inversion to compositional prediction.
    """

    @nn.compact
    def __call__(self, latent, coord_norm, chi_local, rho_local):
        x = jnp.concatenate([latent, coord_norm,
                             jnp.atleast_1d(chi_local * 100),
                             jnp.atleast_1d(rho_local * 10)])
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        return nn.Dense(3)(x)


# NOTE: We deliberately do NOT add decoders for hyperspectral and geochemistry.
# These surface observations are encoder inputs only. Adding decoder losses
# on them creates a parasitic autoencoder shortcut: the encoder sees the data
# and a decoder reconstructs that same data → trivially minimized without
# learning anything about the subsurface. The latent space must be shaped
# by SUBSURFACE targets (forward models + borehole compositions).

# %% [markdown]
# ---
# # Part 5: Multi-Physics Loss Function
#
# Loss targets are SUBSURFACE quantities only:
# - **Magnetics / Gravity**: physics-based forward model losses (G @ property)
# - **Drill holes**: ILR composition reconstruction loss
#
# No explicit smoothness/smallness regularization is needed: the
# coordinate-conditioned MLP decoders share weights across all cells,
# providing implicit spatial regularization (similar to NeRF).

# %%
def loss_fn(params, encoder, dec_susc, dec_dens, dec_ilr,
            mag_grid, grav_grid, hyper_img, geochem_grid,
            mag_observed, grav_observed,
            all_coords_norm, train_coords_norm, train_ilr, train_cell_idx,
            rng_key,
            lambda_recon=10.0, lambda_mag=1.0, lambda_grav=0.3):
    """
    Multi-physics loss with subsurface-only targets:
    - Magnetic forward:    ||G_mag @ χ - mag_obs||² / var      (physics)
    - Gravity forward:     ||G_grav @ Δρ - grav_obs||² / var   (physics)
    - ILR reconstruction:  ||dec_ilr(z,χ,ρ,xyz) - ilr_obs||²   (borehole)

    Surface data (hyperspectral, geochemistry) are encoder inputs only —
    NO decoder losses on them (avoids parasitic autoencoder shortcuts).
    Deterministic encoder (no VAE sampling / KL) — avoids KL fighting recon.
    """

    # --- Encode (all 4 surface sensors) → deterministic latent ---
    mu, log_var = encoder.apply(params['encoder'],
                                mag_grid, grav_grid, hyper_img, geochem_grid)
    z = mu  # deterministic — no sampling, no KL

    # --- Decode physical properties at ALL cell locations ---
    chi_pred = vmap(lambda c: dec_susc.apply(params['dec_susc'], z, c))(all_coords_norm)
    rho_pred = vmap(lambda c: dec_dens.apply(params['dec_dens'], z, c))(all_coords_norm)

    # --- Forward models (PINN) — geophysical consistency ---
    mag_pred = forward_magnetic_jax(chi_pred)
    grav_pred = forward_gravity_jax(rho_pred)

    loss_mag = jnp.mean((mag_pred - mag_observed)**2) / jnp.var(mag_observed)
    loss_grav = jnp.mean((grav_pred - grav_observed)**2) / jnp.var(grav_observed)

    # --- Decode ILR at drill hole locations ---
    chi_at_train = chi_pred[train_cell_idx]
    rho_at_train = rho_pred[train_cell_idx]

    def predict_ilr(coord, chi_local, rho_local):
        return dec_ilr.apply(params['dec_ilr'], z, coord, chi_local, rho_local)

    ilr_pred = vmap(predict_ilr)(train_coords_norm, chi_at_train, rho_at_train)
    loss_recon = jnp.mean((ilr_pred - train_ilr)**2)

    # --- Total ---
    loss_total = (lambda_recon * loss_recon
                  + lambda_mag * loss_mag
                  + lambda_grav * loss_grav)

    return loss_total, {
        'total': loss_total, 'recon': loss_recon,
        'mag': loss_mag, 'grav': loss_grav,
        'chi_max': jnp.max(chi_pred), 'rho_max': jnp.max(rho_pred)
    }

# %% [markdown]
# ---
# # Part 6: Training

# %%
# Initialize all modules
rng = random.PRNGKey(42)
rng, *init_rngs = random.split(rng, 5)

latent_dim = 32
encoder = MultiModalEncoder(latent_dim=latent_dim)
dec_susc = DecoderSusceptibility()
dec_dens = DecoderDensity()
dec_ilr = DecoderILR()

# Dummy inputs for initialization
dummy_mag = jnp.zeros((n_stations_mag, n_stations_mag))
dummy_grav = jnp.zeros((n_stations_mag, n_stations_mag))
dummy_hyper = jnp.zeros((ny, nx, n_bands))
dummy_geochem = jnp.zeros((ny, nx, 5))
dummy_latent = jnp.zeros(latent_dim)
dummy_coord = jnp.zeros(3)
dummy_chi = jnp.float32(0.0)
dummy_rho = jnp.float32(0.0)

params = {
    'encoder': encoder.init(init_rngs[0], dummy_mag, dummy_grav, dummy_hyper, dummy_geochem),
    'dec_susc': dec_susc.init(init_rngs[1], dummy_latent, dummy_coord),
    'dec_dens': dec_dens.init(init_rngs[2], dummy_latent, dummy_coord),
    'dec_ilr': dec_ilr.init(init_rngs[3], dummy_latent, dummy_coord, dummy_chi, dummy_rho),
}

# %%
# Prepare data tensors
mag_grid = jnp.array(mag_observed.reshape(n_stations_mag, n_stations_mag))
mag_grid_norm = (mag_grid - mag_grid.mean()) / (mag_grid.std() + 1e-6)

grav_grid = jnp.array(grav_observed.reshape(n_stations_mag, n_stations_mag))
grav_grid_norm = (grav_grid - grav_grid.mean()) / (grav_grid.std() + 1e-6)

hyper_img_jax = jnp.array(hyper_image)
geochem_grid_jax = jnp.array(geochem_grid_2d)

mag_obs_jax = jnp.array(mag_observed)
grav_obs_jax = jnp.array(grav_observed)

# Coordinates (normalized) — for all cells and drill holes
coord_mean = jnp.array([nx*dx/2, ny*dy/2, -nz*dz/2])
coord_std = jnp.array([nx*dx/2, ny*dy/2, nz*dz/2])
all_coords_norm = jnp.array((cc - np.array(coord_mean)) / np.array(coord_std))
train_coords_norm = jnp.array((df_train[['x', 'y', 'z']].values - np.array(coord_mean)) / np.array(coord_std))
train_ilr = jnp.array(df_train[['ilr0', 'ilr1', 'ilr2']].values)
train_cell_idx = jnp.array(df_train['cell_idx'].values, dtype=jnp.int32)

val_coords_norm = jnp.array((df_val[['x', 'y', 'z']].values - np.array(coord_mean)) / np.array(coord_std))
val_ilr = jnp.array(df_val[['ilr0', 'ilr1', 'ilr2']].values)
val_cell_idx = jnp.array(df_val['cell_idx'].values, dtype=jnp.int32)

print("Data prepared for training")

# %%
# Optimizer with warmup cosine decay
n_epochs = 12000
schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4,
    peak_value=2e-3,
    warmup_steps=500,
    decay_steps=n_epochs
)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(params)

@jit
def train_step(params, opt_state, rng_key):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, encoder, dec_susc, dec_dens, dec_ilr,
        mag_grid_norm, grav_grid_norm, hyper_img_jax, geochem_grid_jax,
        mag_obs_jax, grav_obs_jax,
        all_coords_norm, train_coords_norm, train_ilr, train_cell_idx,
        rng_key,
        lambda_recon=10.0,
        lambda_mag=1.0,
        lambda_grav=0.3
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# %%
# Training loop
history = []

for epoch in range(n_epochs):
    rng, step_rng = random.split(rng)

    params, opt_state, loss, aux = train_step(params, opt_state, step_rng)
    history.append({**{k: float(v) for k, v in aux.items()}, 'epoch': epoch})

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | loss={aux['total']:.4f} | recon={aux['recon']:.4f} | "
              f"mag={aux['mag']:.4f} | grav={aux['grav']:.4f} | "
              f"χ_max={aux['chi_max']:.4f} | ρ_max={aux['rho_max']:.4f}")

# %%
# Training curves
df_hist = pd.DataFrame(history)

(
    lp.ggplot(df_hist, lp.aes(x='epoch'))
    + lp.geom_line(lp.aes(y='recon'), color="#962561")
    + lp.geom_line(lp.aes(y='mag'), color="#abc000")
    + lp.geom_line(lp.aes(y='grav'), color="#006080")
    + lp.scale_y_log10()
    + lp.labs(title='Training losses', x='Epoch', y='Loss (log)')
)

# %% [markdown]
# ---
# # Part 7: Predictions

# %%
# Encode → latent (deterministic, using mean)
mu, log_var = encoder.apply(params['encoder'],
                            mag_grid_norm, grav_grid_norm,
                            hyper_img_jax, geochem_grid_jax)

# Decode physical properties at all cell locations
chi_pred = vmap(lambda c: dec_susc.apply(params['dec_susc'], mu, c))(all_coords_norm)
rho_pred = vmap(lambda c: dec_dens.apply(params['dec_dens'], mu, c))(all_coords_norm)
chi_pred_np = np.array(chi_pred)
rho_pred_np = np.array(rho_pred)

print(f"χ predicted: {chi_pred_np.min():.4f} – {chi_pred_np.max():.4f} SI")
print(f"χ true:      {susceptibility_true.min():.4f} – {susceptibility_true.max():.4f} SI")
print(f"Δρ predicted: {rho_pred_np.min():.3f} – {rho_pred_np.max():.3f} g/cc")
print(f"Δρ true:      {density_true.min():.3f} – {density_true.max():.3f} g/cc")

# %%
# Verify forward models
mag_from_pred = forward_magnetic_jax(chi_pred)
grav_from_pred = forward_gravity_jax(rho_pred)
print(f"Mag predicted: {float(mag_from_pred.min()):.1f} – {float(mag_from_pred.max()):.1f} nT")
print(f"Mag observed:  {mag_observed.min():.1f} – {mag_observed.max():.1f} nT")
print(f"Grav predicted: {float(grav_from_pred.min()):.4f} – {float(grav_from_pred.max()):.4f} mGal")
print(f"Grav observed:  {grav_observed.min():.4f} – {grav_observed.max():.4f} mGal")

# %% [markdown]
# ## Predict compositions with uncertainty (latent perturbation)

# %%
# Since we use deterministic encoding, we estimate uncertainty
# by perturbing the latent vector with small noise.
n_samples = 100
ilr_samples = []
perturbation_scale = 0.1

for i in range(n_samples):
    rng, sample_rng = random.split(rng)
    eps = random.normal(sample_rng, mu.shape)
    z = mu + perturbation_scale * eps

    chi_sample = vmap(lambda c: dec_susc.apply(params['dec_susc'], z, c))(all_coords_norm)
    rho_sample = vmap(lambda c: dec_dens.apply(params['dec_dens'], z, c))(all_coords_norm)

    def predict_ilr(coord, chi_local, rho_local):
        return dec_ilr.apply(params['dec_ilr'], z, coord, chi_local, rho_local)

    ilr_sample = vmap(predict_ilr)(all_coords_norm, chi_sample, rho_sample)
    ilr_samples.append(np.array(ilr_sample))

ilr_samples = np.stack(ilr_samples)
ilr_mean = ilr_samples.mean(axis=0)
ilr_std = ilr_samples.std(axis=0)

print(f"ILR predicted: {ilr_mean.min():.2f} – {ilr_mean.max():.2f}")
print(f"Uncertainty:   {ilr_std.min():.3f} – {ilr_std.max():.3f}")

# %%
comp_pred = ilr_inverse(ilr_mean)
Cu_pred = comp_pred[:, 0]

print(f"Cu predicted: {Cu_pred.min()*100:.2f}% – {Cu_pred.max()*100:.2f}%")
print(f"Cu true:      {Cu.min()*100:.2f}% – {Cu.max()*100:.2f}%")

# %% [markdown]
# ## Validation Metrics

# %%
def predict_ilr_at_coords(mu_latent, coords_norm, cell_idx):
    chi_at = chi_pred[cell_idx]
    rho_at = rho_pred[cell_idx]
    def predict(coord, chi_l, rho_l):
        return dec_ilr.apply(params['dec_ilr'], mu_latent, coord, chi_l, rho_l)
    return vmap(predict)(coords_norm, chi_at, rho_at)

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

# Density metrics
rho_rmse = float(np.sqrt(np.mean((rho_pred_np - density_true)**2)))
ss_res_rho = float(np.sum((rho_pred_np - density_true)**2))
ss_tot_rho = float(np.sum((density_true - density_true.mean())**2))
rho_r2 = 1 - ss_res_rho / ss_tot_rho

print(f"--- Validation Metrics ---")
print(f"ILR Train  RMSE: {train_rmse:.4f} | R²: {train_r2:.4f}")
print(f"ILR Val    RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
print(f"χ global   RMSE: {chi_rmse:.6f} | R²: {chi_r2:.4f}")
print(f"Δρ global  RMSE: {rho_rmse:.4f} | R²: {rho_r2:.4f}")

# %% [markdown]
# ---
# # Part 8: Visualizations

# %%
# Horizontal slice at mid-depth
iz = nz // 2
idx_slice = slice(iz * nx * ny, (iz + 1) * nx * ny)

df_results = pd.DataFrame({
    'x': cc[idx_slice, 0],
    'y': cc[idx_slice, 1],
    'chi_pred': chi_pred_np[idx_slice],
    'chi_true': susceptibility_true[idx_slice],
    'rho_pred': rho_pred_np[idx_slice],
    'rho_true': density_true[idx_slice],
    'Cu_pred': Cu_pred[idx_slice] * 100,
    'Cu_true': Cu[idx_slice] * 100,
    'uncertainty': ilr_std[idx_slice, 0]
})

df_holes_train = df_train.drop_duplicates('hole')[['x', 'y']]
df_holes_val = df_val.drop_duplicates('hole')[['x', 'y']]

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='chi_pred'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkblue', name='χ (SI)')
    + lp.labs(title=f'Predicted susceptibility (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='chi_true'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkblue', name='χ (SI)')
    + lp.labs(title=f'True susceptibility (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='rho_pred'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='#2d1b69', name='Δρ (g/cc)')
    + lp.labs(title=f'Predicted density contrast (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='rho_true'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='#2d1b69', name='Δρ (g/cc)')
    + lp.labs(title=f'True density contrast (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='Cu_pred'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkred', name='Cu %')
    + lp.labs(title=f'Predicted Cu (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='Cu_true'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkred', name='Cu %')
    + lp.labs(title=f'True Cu (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='uncertainty'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='purple', name='σ')
    + lp.labs(title=f'Uncertainty on ILR[0] (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %% [markdown]
# ## Quantitative Comparisons

# %%
(
    lp.ggplot(df_results, lp.aes('chi_true', 'chi_pred'))
    + lp.geom_point(alpha=0.5)
    + lp.geom_abline(slope=1, intercept=0, color='red', linetype='dashed')
    + lp.labs(title='Susceptibility: predicted vs true', x='χ true', y='χ predicted')
)

# %%
(
    lp.ggplot(df_results, lp.aes('rho_true', 'rho_pred'))
    + lp.geom_point(alpha=0.5)
    + lp.geom_abline(slope=1, intercept=0, color='red', linetype='dashed')
    + lp.labs(title='Density contrast: predicted vs true', x='Δρ true (g/cc)', y='Δρ predicted (g/cc)')
)

# %%
(
    lp.ggplot(df_results, lp.aes('Cu_true', 'Cu_pred'))
    + lp.geom_point(alpha=0.5)
    + lp.geom_abline(slope=1, intercept=0, color='red', linetype='dashed')
    + lp.labs(title='Cu: predicted vs true', x='Cu true (%)', y='Cu predicted (%)')
)

# %% [markdown]
# ---
# # Summary
#
# ## Multi-Sensor Data Fusion for Mineral Exploration
#
# This exercise demonstrates a KoBold Metals-style approach to exploration
# data integration, where multiple heterogeneous data sources are fused
# through a shared latent space for joint geophysical-geochemical inversion.
#
# ### Data Integration
#
# | Source           | Information               | Role           | Loss type                |
# |------------------|---------------------------|----------------|--------------------------|
# | Airborne mag     | Magnetic susceptibility   | Encoder input  | Physics: G_mag @ χ       |
# | Ground gravity   | Density contrast          | Encoder input  | Physics: G_grav @ Δρ     |
# | Hyperspectral    | Surface alteration        | Encoder input  | (none — input only)      |
# | Soil geochemistry| Pathfinder elements       | Encoder input  | (none — input only)      |
# | Drill holes      | Direct Cu, Ni, Co assays  | Target only    | Decoder: z+χ+Δρ → ILR   |
#
# ### Key Design Choices
#
# 1. **Multi-branch encoder**: Each sensor has its own CNN branch,
#    allowing the network to learn modality-specific features before
#    fusion. This mirrors KoBold's approach of specialized processors
#    feeding into a unified data platform (TerraShed).
#
# 2. **Joint inversion via shared latent space**: Susceptibility and
#    density are decoded from the same latent vector, enforcing
#    cross-property consistency. The ILR decoder receives both χ and Δρ
#    as inputs, creating a direct physical bridge between geophysical
#    properties and mineral compositions.
#
# 3. **Surface inputs, subsurface targets**: Hyperspectral and geochemistry
#    data enter through the encoder but have NO decoder losses. Adding
#    decoder losses on encoder inputs creates a parasitic autoencoder
#    shortcut: the network trivially memorizes surface data without
#    learning subsurface structure. Only subsurface quantities (forward
#    model predictions, borehole compositions) serve as training targets.
#
# 4. **Coordinate-conditioned decoders (NeRF-style)**: Rather than
#    decoding z → all 2048 cells at once, each cell is predicted from
#    (z, x, y, z_coord). Weight sharing across cells provides implicit
#    spatial regularization, and the network can learn position-dependent
#    anomaly patterns. This approach resolves the non-uniqueness of
#    geophysical inversion by integrating spatial structure.
#
# 5. **Uncertainty quantification**: Latent perturbation sampling
#    propagates uncertainty through all decoders, providing
#    spatially-varying confidence estimates for exploration targeting.
#
# ### Comparison to KoBold Metals' Approach
#
# | KoBold                          | This Exercise                     |
# |---------------------------------|-----------------------------------|
# | TerraShed (data platform)       | Multi-branch encoder              |
# | Machine Prospector (ML engine)  | VAE + multi-physics decoder       |
# | Full-physics joint inversion    | G_mag @ χ + G_grav @ Δρ losses    |
# | Helicopter EM / SQUID           | (could add EM forward model)      |
# | Hyperpod (spectral + LiDAR)     | Hyperspectral CNN branch          |
# | Geochemical analysis            | Soil geochem + drill hole assays  |
# | Uncertainty quantification      | Posterior sampling (100 draws)    |
#
# ### References
#
# - KoBold Metals. AI-powered mineral exploration.
#   IEEE Spectrum (2024): https://spectrum.ieee.org/ai-mining
# - Eagar, K. (SimPEG). Open-source geophysical modeling.
