# %% [markdown]
# # Multi-Sensor Geophysical Inversion with GemPy + NeRF Decoders
#
# ## Architecture: Geological Structure meets Data-Driven Inversion
#
# This exercise combines:
# 1. **GemPy** — implicit 3D geological modeling from structural observations
# 2. **NeRF-style decoders** — coordinate-conditioned property prediction
# 3. **SimPEG** — physics-based forward models (magnetics, gravity)
# 4. **MC Dropout** — Bayesian uncertainty quantification
#
# The key idea: GemPy provides the geological *structure* (where are the
# unit boundaries), while the NeRF decoders predict continuous *property
# values* conditioned on that structure. The GemPy scalar field becomes
# an additional input to each decoder, telling it "how deep are you in
# this geological unit" and "how close are you to a boundary."
#
# ```
#              ┌──────────────────────────────────────────────┐
#              │              GemPy Structural Model          │
#              │  contact points + orientations → φ(x,y,z)   │
#              └──────────────┬───────────────────────────────┘
#                             │ scalar field φ + unit ID
#                             ▼
#              ┌──────────────┐
# Magnetics ──▶│ CNN Branch 1 │──┐
#              └──────────────┘  │
#              ┌──────────────┐  │   ┌─────────┐
# Gravity ────▶│ CNN Branch 2 │──┼──▶│ FUSION  │──▶ Latent z
#              └──────────────┘  │   │  MLP    │       │
#              ┌──────────────┐  │   └─────────┘       │
# Hyperspect. ▶│ CNN Branch 3 │──┤                     │
#              └──────────────┘  │     For each cell:   │
#              ┌──────────────┐  │  ┌──────────────────────────────────┐
# Geochemistry▶│ CNN Branch 4 │──┘  │  (z, coord, φ(coord), unit_id)  │
#              └──────────────┘     │  ┌────────┐  ┌────────┐         │
#                                   │  │DEC_χ   │  │DEC_ρ   │         │
#                                   │  └───┬────┘  └───┬────┘         │
#                                   │      ▼           ▼              │
#                                   │  ┌────────────────────────────┐ │
#                                   │  │     DECODER_ILR            │ │
#                                   │  │ z+coord+φ+unit+χ+ρ → ILR  │ │
#                                   │  └────────────┬───────────────┘ │
#                                   └───────────────┼─────────────────┘
#                          ┌────────────┬───────────┘
#                          ▼            ▼
#                   ┌────────────┐ ┌────────────┐  ┌──────────────┐
#                   │ G_mag @ χ  │ │ G_grav @ Δρ│  │Loss reconstr.│
#                   └─────┬──────┘ └─────┬──────┘  └──────────────┘
#                         ▼              ▼
#                   Loss magnetics  Loss gravity
# ```
#
# **Key design choices**:
# 1. Surface sensors are encoder inputs only — no decoder losses on them.
# 2. GemPy scalar field φ provides geological structure to decoders.
# 3. MC Dropout in decoders for Bayesian uncertainty (Gal & Ghahramani, 2016).

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata

import gempy as gp
from gempy.core.data import StructuralFrame

import lets_plot as lp
lp.LetsPlot.setup_html()

device = torch.device('cpu')
torch.manual_seed(42)

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
# # Part 1b: Borehole Lithological Logs
#
# In real exploration, a geologist logs each drill core interval with a
# lithological classification. We generate these from our synthetic model
# by thresholding the body intensities.

# %%
# Define lithologies based on mineralization intensity
# In reality, a geologist inspects the core and assigns these
def assign_lithology(body1_val, body2_val):
    """Assign lithological class based on ore body proximity."""
    intensity = max(body1_val, body2_val)
    if intensity > 0.5:
        return 'massive_sulfide'
    elif intensity > 0.15:
        return 'disseminated_sulfide'
    elif intensity > 0.05:
        return 'altered_gneiss'
    else:
        return 'fresh_gneiss'

lithology_ids = {'fresh_gneiss': 0, 'altered_gneiss': 1,
                 'disseminated_sulfide': 2, 'massive_sulfide': 3}
n_lith = len(lithology_ids)

# Assign lithology to every cell (for ground truth)
cell_lithology = np.array([assign_lithology(body1[i], body2[i]) for i in range(n_cells)])
cell_lith_id = np.array([lithology_ids[l] for l in cell_lithology])

print("Lithology counts:")
for name, lid in lithology_ids.items():
    count = (cell_lith_id == lid).sum()
    print(f"  {name:25s}: {count:4d} cells ({count/n_cells*100:.1f}%)")

# %% [markdown]
# ---
# # Part 2: Multi-Sensor Surveys

# %% [markdown]
# ## 2a. Airborne Magnetic Survey (TMI)

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
G_mag_np = sim_mag.G
G_mag = torch.tensor(G_mag_np, dtype=torch.float32)

print(f"Magnetic TMI: {mag_observed.min():.1f} – {mag_observed.max():.1f} nT")
print(f"G_mag shape: {G_mag.shape}")

# %% [markdown]
# ## 2b. Ground Gravity Survey (gz)

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
G_grav_np = sim_grav.G
G_grav = torch.tensor(G_grav_np, dtype=torch.float32)

print(f"Gravity gz: {grav_observed.min():.3f} – {grav_observed.max():.3f} mGal")

# %% [markdown]
# ## 2c. Hyperspectral Drone Survey

# %%
surface_xx, surface_yy = np.meshgrid(mesh.cell_centers_x, mesh.cell_centers_y)

d1_surf = np.sqrt(((surface_xx - 1500)/600)**2 + ((surface_yy - 1500)/550)**2)
body1_surf = np.exp(-d1_surf**2 * 1.5) * 0.4

d2_surf = np.sqrt(((surface_xx - 2800)/550)**2 + ((surface_yy - 2500)/500)**2)
body2_surf = np.exp(-d2_surf**2 * 1.5) * 0.25

n_bands = 10
endmember_bg = np.array([0.30, 0.32, 0.35, 0.33, 0.30, 0.28, 0.25, 0.27, 0.30, 0.28])
endmember_fe = np.array([0.15, 0.22, 0.38, 0.55, 0.68, 0.62, 0.48, 0.42, 0.38, 0.33])
endmember_clay = np.array([0.40, 0.45, 0.50, 0.52, 0.58, 0.50, 0.40, 0.28, 0.18, 0.22])

frac_fe = body1_surf + 0.5 * body2_surf
frac_clay = 0.5 * body1_surf + body2_surf
frac_bg = np.clip(1.0 - frac_fe - frac_clay, 0.1, 1.0)
total = frac_bg + frac_fe + frac_clay
frac_bg /= total; frac_fe /= total; frac_clay /= total

hyper_image = (frac_bg[:,:,None] * endmember_bg +
               frac_fe[:,:,None] * endmember_fe +
               frac_clay[:,:,None] * endmember_clay)
hyper_image += rng_np.normal(0, 0.02, hyper_image.shape)
hyper_image = np.clip(hyper_image, 0, 1)

print(f"Hyperspectral image: {hyper_image.shape}")

# %% [markdown]
# ## 2d. Surface Geochemistry

# %%
n_soil = 50
soil_x = rng_np.uniform(200, nx*dx-200, n_soil)
soil_y = rng_np.uniform(200, ny*dy-200, n_soil)

d1_soil = np.sqrt(((soil_x - 1500)/600)**2 + ((soil_y - 1500)/550)**2)
body1_soil = np.exp(-d1_soil**2 * 1.5) * 0.4
d2_soil = np.sqrt(((soil_x - 2800)/550)**2 + ((soil_y - 2500)/500)**2)
body2_soil = np.exp(-d2_soil**2 * 1.5) * 0.25

Cu_soil = np.clip(30 + body1_soil * 500 + body2_soil * 300 + rng_np.normal(0, 10, n_soil), 5, None)
Ni_soil = np.clip(20 + body1_soil * 200 + body2_soil * 600 + rng_np.normal(0, 8, n_soil), 5, None)
Co_soil = np.clip(8 + body1_soil * 80 + body2_soil * 200 + rng_np.normal(0, 4, n_soil), 2, None)
Fe_soil = 3.0 + body1_soil * 5 + body2_soil * 3 + rng_np.normal(0, 0.3, n_soil)
S_soil = 0.1 + body1_soil * 2 + body2_soil * 1.5 + rng_np.normal(0, 0.05, n_soil)

grid_xy = np.column_stack([surface_xx.ravel(), surface_yy.ravel()])
soil_points = np.column_stack([soil_x, soil_y])

geochem_grid = np.zeros((nx * ny, 5))
for i, vals in enumerate([Cu_soil, Ni_soil, Co_soil, Fe_soil, S_soil]):
    grid_lin = griddata(soil_points, vals, grid_xy, method='linear')
    grid_near = griddata(soil_points, vals, grid_xy, method='nearest')
    geochem_grid[:, i] = np.where(np.isnan(grid_lin), grid_near, grid_lin)

geochem_min = geochem_grid.min(axis=0)
geochem_max = geochem_grid.max(axis=0)
geochem_grid_norm = (geochem_grid - geochem_min) / (geochem_max - geochem_min + 1e-8)
geochem_grid_2d = geochem_grid_norm.reshape(ny, nx, 5)

# %% [markdown]
# ## 2e. Drill Holes with Lithological Logs

# %%
hole_xy = [
    (1500, 1500), (2800, 2500), (500, 500), (2000, 2000), (3500, 1000),
    (1200, 1800), (2500, 2200), (1800, 2800), (3000, 1500), (800, 2500)
]

validation_holes = {'DH4', 'DH7'}
noise_rel = 0.05

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
        lith = cell_lithology[idx]
        forages.append({
            'hole': hole_name,
            'x': mesh.cell_centers_x[ix], 'y': mesh.cell_centers_y[iy],
            'z': mesh.cell_centers_z[iz],
            'Cu': cu_noisy, 'Ni': ni_noisy, 'Co': co_noisy,
            'ilr0': ilr_noisy[0], 'ilr1': ilr_noisy[1], 'ilr2': ilr_noisy[2],
            'lithology': lith,
            'lith_id': lithology_ids[lith],
            'cell_idx': idx,
            'is_validation': hole_name in validation_holes
        })

df_forages = pd.DataFrame(forages)
df_train = df_forages[~df_forages['is_validation']].reset_index(drop=True)
df_val = df_forages[df_forages['is_validation']].reset_index(drop=True)

print(f"Drill samples: {len(df_forages)} | train: {len(df_train)} | val: {len(df_val)}")
print(f"\nLithology log (DH1):")
dh1 = df_forages[df_forages['hole'] == 'DH1'][['z', 'lithology', 'Cu']].reset_index(drop=True)
for _, row in dh1.iterrows():
    print(f"  z={row['z']:7.0f}m  {row['lithology']:25s}  Cu={row['Cu']*100:.2f}%")

# %% [markdown]
# ---
# # Part 2f: GemPy Structural Model
#
# We build a GemPy geological model from the drill hole lithological logs.
# In a real project, a geologist would pick contact points where lithology
# changes in core, and estimate orientations from multiple holes. Here we
# extract these automatically from our synthetic borehole data.
#
# GemPy interpolates a scalar potential field φ(x,y,z) that defines
# the geological unit boundaries. This scalar field becomes an input
# to our NeRF decoders — it tells the network WHERE geological
# boundaries are, so the network can focus on WHAT properties exist
# within each unit.

# %%
# Extract contact points from drill holes: locations where lithology changes
contacts = []
orientations_data = []

for hole_name in df_forages['hole'].unique():
    dh = df_forages[df_forages['hole'] == hole_name].sort_values('z', ascending=False)
    rows = dh.reset_index(drop=True)

    for j in range(len(rows) - 1):
        if rows.loc[j, 'lith_id'] != rows.loc[j+1, 'lith_id']:
            # Contact midpoint between the two intervals
            cx = float(rows.loc[j, 'x'])
            cy = float(rows.loc[j, 'y'])
            cz = float((rows.loc[j, 'z'] + rows.loc[j+1, 'z']) / 2)
            contacts.append({
                'x': cx, 'y': cy, 'z': cz,
                'surface': 'ore_contact'
            })

# Need at least a few points. Add surface reference points (host rock at surface)
for x_ref in [500, 2000, 3500]:
    for y_ref in [500, 2000, 3500]:
        contacts.append({'x': float(x_ref), 'y': float(y_ref), 'z': -62.5,
                         'surface': 'surface_ref'})

df_contacts = pd.DataFrame(contacts)
print(f"Contact points extracted: {len(df_contacts)}")
print(f"  ore_contact: {(df_contacts['surface']=='ore_contact').sum()}")
print(f"  surface_ref: {(df_contacts['surface']=='surface_ref').sum()}")

# %%
# Build GemPy model
geo_model = gp.create_geomodel(
    project_name='sulfide_exploration',
    extent=[0, 4000, 0, 4000, -1000, 0],
    resolution=[16, 16, 8],
    structural_frame=StructuralFrame.initialize_default_structure()
)

# Add surface contact points
ore_contacts = df_contacts[df_contacts['surface'] == 'ore_contact']
if len(ore_contacts) > 0:
    gp.add_surface_points(
        geo_model,
        x=ore_contacts['x'].tolist(),
        y=ore_contacts['y'].tolist(),
        z=ore_contacts['z'].tolist(),
        elements_names=['surface1'] * len(ore_contacts)
    )

# Add orientations (vertical for our case — bodies are roughly horizontal lenses)
# In reality, a geologist would estimate these from core and structural measurements
gp.add_orientations(
    geo_model,
    x=[1500.0, 2800.0],
    y=[1500.0, 2500.0],
    z=[-500.0, -700.0],
    elements_names=['surface1', 'surface1'],
    pole_vector=[[0, 0, 1], [0, 0, 1]]
)

geo_model.update_transform()

# Compute structural model
print("Computing GemPy structural model...")
sol = gp.compute_model(geo_model)
print("GemPy model computed.")

# %%
# Extract scalar field at all cell centers using custom grid
gp.set_custom_grid(geo_model.grid, cc.astype(np.float64))
sol = gp.compute_model(geo_model)

# Get scalar field values from the octree output
oo0 = sol.octrees_output[0]
gc0 = oo0.grid_centers
oc0 = oo0.output_centers
ef0 = oc0.exported_fields

custom_slice = gc0.custom_grid_slice
scalar_field_at_cells = np.array(ef0.scalar_field[custom_slice])

# Normalize scalar field to [0, 1] for use as network input
sf_min, sf_max = scalar_field_at_cells.min(), scalar_field_at_cells.max()
if sf_max > sf_min:
    scalar_field_norm = (scalar_field_at_cells - sf_min) / (sf_max - sf_min)
else:
    scalar_field_norm = np.zeros_like(scalar_field_at_cells)

print(f"Scalar field: {scalar_field_at_cells.min():.4f} – {scalar_field_at_cells.max():.4f}")
print(f"Normalized:   {scalar_field_norm.min():.4f} – {scalar_field_norm.max():.4f}")

# Get lithology IDs from GemPy
gempy_lith_ids = np.array(oc0.ids_custom_grid, dtype=np.int32)
print(f"GemPy lithology IDs: unique = {np.unique(gempy_lith_ids)}")

# %% [markdown]
# ---
# # Part 3: Differentiable Forward Models

# %%
def forward_magnetic(susceptibility):
    """Forward magnetics: TMI = G_mag @ χ"""
    return G_mag @ susceptibility

def forward_gravity(density):
    """Forward gravity: gz = G_grav @ Δρ"""
    return G_grav @ density

# %% [markdown]
# ---
# # Part 4: Network Architecture (PyTorch)
#
# Multi-modal encoder fuses 4 sensor types into a latent vector.
# NeRF-style decoders take (z, coord, φ_gempy, unit_embedding) → properties.
# MC Dropout in decoders for Bayesian uncertainty.

# %%
class MultiModalEncoder(nn.Module):
    """
    Fuses magnetic, gravity, hyperspectral, and geochemistry inputs
    into a shared latent vector.
    """
    def __init__(self, latent_dim=32, n_mag=20, n_hyper_bands=10):
        super().__init__()
        self.latent_dim = latent_dim

        # Magnetic branch: (1, 20, 20) → 64
        self.mag_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.mag_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.mag_fc = nn.Linear(32 * n_mag * n_mag, 64)

        # Gravity branch: (1, 20, 20) → 64
        self.grav_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.grav_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.grav_fc = nn.Linear(32 * n_mag * n_mag, 64)

        # Hyperspectral branch: (10, 16, 16) → 64
        self.hyper_conv1 = nn.Conv2d(n_hyper_bands, 16, 3, padding=1)
        self.hyper_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.hyper_fc = nn.Linear(32 * 16 * 16, 64)

        # Geochemistry branch: (5, 16, 16) → 32
        self.geochem_conv = nn.Conv2d(5, 16, 3, padding=1)
        self.geochem_fc = nn.Linear(16 * 16 * 16, 32)

        # Fusion MLP: 224 → 128 → 64 → latent
        self.fusion = nn.Sequential(
            nn.Linear(224, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.mu_head = nn.Linear(64, latent_dim)
        self.logvar_head = nn.Linear(64, latent_dim)

    def forward(self, mag_grid, grav_grid, hyper_img, geochem_grid):
        # mag_grid: (20, 20) → (1, 1, 20, 20)
        m = F.relu(self.mag_conv1(mag_grid.unsqueeze(0).unsqueeze(0)))
        m = F.relu(self.mag_conv2(m))
        m = F.relu(self.mag_fc(m.flatten()))

        g = F.relu(self.grav_conv1(grav_grid.unsqueeze(0).unsqueeze(0)))
        g = F.relu(self.grav_conv2(g))
        g = F.relu(self.grav_fc(g.flatten()))

        # hyper_img: (16, 16, 10) → (1, 10, 16, 16)
        h = hyper_img.permute(2, 0, 1).unsqueeze(0)
        h = F.relu(self.hyper_conv1(h))
        h = F.relu(self.hyper_conv2(h))
        h = F.relu(self.hyper_fc(h.flatten()))

        # geochem: (16, 16, 5) → (1, 5, 16, 16)
        c = geochem_grid.permute(2, 0, 1).unsqueeze(0)
        c = F.relu(self.geochem_conv(c))
        c = F.relu(self.geochem_fc(c.flatten()))

        fused = torch.cat([m, g, h, c])  # 64+64+64+32 = 224
        fused = self.fusion(fused)
        return self.mu_head(fused), self.logvar_head(fused)


class DecoderSusceptibility(nn.Module):
    """
    (latent, coord_norm, φ_gempy, unit_embedding) → χ_local.
    GemPy scalar field provides geological structure awareness.
    MC Dropout for Bayesian uncertainty.
    """
    def __init__(self, latent_dim=32, n_units=4, embed_dim=4, dropout=0.1):
        super().__init__()
        self.unit_embed = nn.Embedding(n_units, embed_dim)
        input_dim = latent_dim + 3 + 1 + embed_dim  # z + coord + φ + unit_emb
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, latent, coord, phi, unit_id):
        """
        latent: (latent_dim,) or (N, latent_dim)
        coord: (3,) or (N, 3)
        phi: scalar or (N,)
        unit_id: int or (N,) long tensor
        """
        u_emb = self.unit_embed(unit_id)
        if phi.dim() == 0:
            phi = phi.unsqueeze(0)
        if phi.dim() == 1 and latent.dim() == 1:
            x = torch.cat([latent, coord, phi, u_emb])
        else:
            x = torch.cat([latent, coord, phi.unsqueeze(-1) if phi.dim() == 1 else phi, u_emb], dim=-1)
        return F.softplus(self.net(x).squeeze(-1)) * 0.01


class DecoderDensity(nn.Module):
    """(latent, coord_norm, φ_gempy, unit_embedding) → Δρ_local."""
    def __init__(self, latent_dim=32, n_units=4, embed_dim=4, dropout=0.1):
        super().__init__()
        self.unit_embed = nn.Embedding(n_units, embed_dim)
        input_dim = latent_dim + 3 + 1 + embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, latent, coord, phi, unit_id):
        u_emb = self.unit_embed(unit_id)
        if phi.dim() == 0:
            phi = phi.unsqueeze(0)
        if phi.dim() == 1 and latent.dim() == 1:
            x = torch.cat([latent, coord, phi, u_emb])
        else:
            x = torch.cat([latent, coord, phi.unsqueeze(-1) if phi.dim() == 1 else phi, u_emb], dim=-1)
        return F.softplus(self.net(x).squeeze(-1)) * 0.1


class DecoderILR(nn.Module):
    """
    (latent, coord, φ, unit_id, χ, ρ) → ILR compositions (3-dim).
    Receives physical properties as inputs — bridges geophysics to geochemistry.
    """
    def __init__(self, latent_dim=32, n_units=4, embed_dim=4, dropout=0.1):
        super().__init__()
        self.unit_embed = nn.Embedding(n_units, embed_dim)
        input_dim = latent_dim + 3 + 1 + embed_dim + 2  # + chi_scaled + rho_scaled
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, latent, coord, phi, unit_id, chi_local, rho_local):
        u_emb = self.unit_embed(unit_id)
        if phi.dim() == 0:
            phi = phi.unsqueeze(0)

        # Scale physical properties to similar magnitude as other inputs
        chi_s = (chi_local * 100).unsqueeze(-1) if chi_local.dim() == 1 else (chi_local * 100)
        rho_s = (rho_local * 10).unsqueeze(-1) if rho_local.dim() == 1 else (rho_local * 10)

        if latent.dim() == 1:
            x = torch.cat([latent, coord, phi, u_emb, chi_s, rho_s])
        else:
            x = torch.cat([latent, coord,
                           phi.unsqueeze(-1) if phi.dim() == 1 else phi,
                           u_emb, chi_s, rho_s], dim=-1)
        return self.net(x)


# NOTE: No decoders for hyperspectral/geochemistry — surface observations
# are encoder inputs only, to avoid parasitic autoencoder shortcuts.

# %% [markdown]
# ---
# # Part 5: Multi-Physics Loss Function

# %%
def loss_fn(encoder, dec_susc, dec_dens, dec_ilr,
            mag_grid_t, grav_grid_t, hyper_t, geochem_t,
            mag_obs_t, grav_obs_t,
            all_coords_t, all_phi_t, all_unit_ids_t,
            train_coords_t, train_phi_t, train_unit_ids_t,
            train_ilr_t, train_ilr_std_t, train_cell_idx,
            lambda_recon=10.0, lambda_mag=1.0, lambda_grav=0.3):
    """
    Multi-physics loss with GemPy structural conditioning.
    Dropout is active during training (model.train() mode).
    """
    # --- Encode → deterministic latent ---
    mu, log_var = encoder(mag_grid_t, grav_grid_t, hyper_t, geochem_t)
    z = mu  # deterministic

    # --- Decode at ALL cells (z broadcast to each cell) ---
    z_expanded = z.unsqueeze(0).expand(all_coords_t.shape[0], -1)
    chi_pred = dec_susc(z_expanded, all_coords_t, all_phi_t, all_unit_ids_t)
    rho_pred = dec_dens(z_expanded, all_coords_t, all_phi_t, all_unit_ids_t)

    # --- Forward physics ---
    mag_pred = forward_magnetic(chi_pred)
    grav_pred = forward_gravity(rho_pred)

    loss_mag = torch.mean((mag_pred - mag_obs_t)**2) / torch.var(mag_obs_t)
    loss_grav = torch.mean((grav_pred - grav_obs_t)**2) / torch.var(grav_obs_t)

    # --- Decode ILR at drill holes ---
    chi_at_train = chi_pred[train_cell_idx]
    rho_at_train = rho_pred[train_cell_idx]

    z_train = z.unsqueeze(0).expand(train_coords_t.shape[0], -1)
    ilr_pred = dec_ilr(z_train, train_coords_t, train_phi_t,
                       train_unit_ids_t, chi_at_train, rho_at_train)

    loss_recon = torch.mean(((ilr_pred - train_ilr_t) / train_ilr_std_t)**2)

    # --- Rho bounding penalty (sum-based) ---
    loss_rho_bound = torch.sum(torch.clamp(rho_pred - 0.25, min=0.0)**2)

    # --- Total ---
    loss_total = (lambda_recon * loss_recon
                  + lambda_mag * loss_mag
                  + lambda_grav * loss_grav
                  + 1.0 * loss_rho_bound)

    return loss_total, {
        'total': loss_total.item(), 'recon': loss_recon.item(),
        'mag': loss_mag.item(), 'grav': loss_grav.item(),
        'chi_max': chi_pred.max().item(), 'rho_max': rho_pred.max().item(),
        'rho_bnd': loss_rho_bound.item()
    }

# %% [markdown]
# ---
# # Part 6: Training

# %%
# Initialize models
latent_dim = 32
encoder = MultiModalEncoder(latent_dim=latent_dim)
dec_susc = DecoderSusceptibility(latent_dim=latent_dim, n_units=n_lith)
dec_dens = DecoderDensity(latent_dim=latent_dim, n_units=n_lith)
dec_ilr = DecoderILR(latent_dim=latent_dim, n_units=n_lith)

# Count parameters
n_params = sum(p.numel() for p in encoder.parameters())
n_params += sum(p.numel() for m in [dec_susc, dec_dens, dec_ilr] for p in m.parameters())
print(f"Total parameters: {n_params:,}")

# %%
# Prepare data tensors
mag_grid_t = torch.tensor(mag_observed.reshape(n_stations_mag, n_stations_mag), dtype=torch.float32)
mag_grid_t = (mag_grid_t - mag_grid_t.mean()) / (mag_grid_t.std() + 1e-6)

grav_grid_t = torch.tensor(grav_observed.reshape(n_stations_mag, n_stations_mag), dtype=torch.float32)
grav_grid_t = (grav_grid_t - grav_grid_t.mean()) / (grav_grid_t.std() + 1e-6)

hyper_t = torch.tensor(hyper_image, dtype=torch.float32)
geochem_t = torch.tensor(geochem_grid_2d, dtype=torch.float32)

mag_obs_t = torch.tensor(mag_observed, dtype=torch.float32)
grav_obs_t = torch.tensor(grav_observed, dtype=torch.float32)

# Coordinates (normalized)
coord_mean = np.array([nx*dx/2, ny*dy/2, -nz*dz/2])
coord_std = np.array([nx*dx/2, ny*dy/2, nz*dz/2])
all_coords_t = torch.tensor((cc - coord_mean) / coord_std, dtype=torch.float32)
all_phi_t = torch.tensor(scalar_field_norm, dtype=torch.float32)
all_unit_ids_t = torch.tensor(cell_lith_id, dtype=torch.long)

train_coords_t = torch.tensor((df_train[['x', 'y', 'z']].values - coord_mean) / coord_std, dtype=torch.float32)
train_ilr_t = torch.tensor(df_train[['ilr0', 'ilr1', 'ilr2']].values, dtype=torch.float32)
train_ilr_std_t = train_ilr_t.std(dim=0)
train_cell_idx = torch.tensor(df_train['cell_idx'].values, dtype=torch.long)

# GemPy features for train drill holes
train_phi_t = all_phi_t[train_cell_idx]
train_unit_ids_t = all_unit_ids_t[train_cell_idx]

val_coords_t = torch.tensor((df_val[['x', 'y', 'z']].values - coord_mean) / coord_std, dtype=torch.float32)
val_ilr_t = torch.tensor(df_val[['ilr0', 'ilr1', 'ilr2']].values, dtype=torch.float32)
val_cell_idx = torch.tensor(df_val['cell_idx'].values, dtype=torch.long)

print("Data prepared for training")

# %%
# Optimizer with cosine annealing
n_epochs = 25000
all_params = (list(encoder.parameters()) + list(dec_susc.parameters())
              + list(dec_dens.parameters()) + list(dec_ilr.parameters()))
optimizer = torch.optim.Adam(all_params, lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=n_epochs, eta_min=1e-5)

# Manual warmup for first 500 steps
warmup_steps = 500

# %%
# Training loop
history = []

encoder.train()
dec_susc.train()
dec_dens.train()
dec_ilr.train()

for epoch in range(n_epochs):
    # Warmup: linearly increase LR from 1e-4 to 2e-3
    if epoch < warmup_steps:
        lr = 1e-4 + (2e-3 - 1e-4) * epoch / warmup_steps
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    optimizer.zero_grad()

    loss, aux = loss_fn(
        encoder, dec_susc, dec_dens, dec_ilr,
        mag_grid_t, grav_grid_t, hyper_t, geochem_t,
        mag_obs_t, grav_obs_t,
        all_coords_t, all_phi_t, all_unit_ids_t,
        train_coords_t, train_phi_t, train_unit_ids_t,
        train_ilr_t, train_ilr_std_t, train_cell_idx,
        lambda_recon=10.0, lambda_mag=1.0, lambda_grav=0.3
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
    optimizer.step()

    if epoch >= warmup_steps:
        scheduler.step()

    history.append({**aux, 'epoch': epoch})

    if epoch % 2500 == 0:
        print(f"Epoch {epoch:5d} | loss={aux['total']:.4f} | recon={aux['recon']:.4f} | "
              f"mag={aux['mag']:.4f} | grav={aux['grav']:.4f} | "
              f"χ_max={aux['chi_max']:.5f} | ρ_max={aux['rho_max']:.5f}")

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
# # Part 7: Predictions with MC Dropout Uncertainty

# %%
# Deterministic predictions (dropout off)
encoder.eval()
dec_susc.eval()
dec_dens.eval()
dec_ilr.eval()

with torch.no_grad():
    mu, log_var = encoder(mag_grid_t, grav_grid_t, hyper_t, geochem_t)
    z_expanded = mu.unsqueeze(0).expand(all_coords_t.shape[0], -1)

    chi_pred = dec_susc(z_expanded, all_coords_t, all_phi_t, all_unit_ids_t)
    rho_pred = dec_dens(z_expanded, all_coords_t, all_phi_t, all_unit_ids_t)

chi_pred_np = chi_pred.detach().numpy()
rho_pred_np = rho_pred.detach().numpy()

print(f"χ predicted: {chi_pred_np.min():.4f} – {chi_pred_np.max():.4f} SI")
print(f"χ true:      {susceptibility_true.min():.4f} – {susceptibility_true.max():.4f} SI")
print(f"Δρ predicted: {rho_pred_np.min():.3f} – {rho_pred_np.max():.3f} g/cc")
print(f"Δρ true:      {density_true.min():.3f} – {density_true.max():.3f} g/cc")

# %%
# MC Dropout uncertainty — keep dropout ON during inference
n_mc_samples = 50
chi_samples = []
rho_samples = []
ilr_samples = []

# Enable dropout for MC sampling
dec_susc.train()
dec_dens.train()
dec_ilr.train()

with torch.no_grad():
    mu, _ = encoder(mag_grid_t, grav_grid_t, hyper_t, geochem_t)
    z_expanded = mu.unsqueeze(0).expand(all_coords_t.shape[0], -1)

    for i in range(n_mc_samples):
        chi_s = dec_susc(z_expanded, all_coords_t, all_phi_t, all_unit_ids_t)
        rho_s = dec_dens(z_expanded, all_coords_t, all_phi_t, all_unit_ids_t)
        ilr_s = dec_ilr(z_expanded, all_coords_t, all_phi_t, all_unit_ids_t, chi_s, rho_s)

        chi_samples.append(chi_s.numpy())
        rho_samples.append(rho_s.numpy())
        ilr_samples.append(ilr_s.numpy())

chi_samples = np.stack(chi_samples)
rho_samples = np.stack(rho_samples)
ilr_samples = np.stack(ilr_samples)

ilr_mean = ilr_samples.mean(axis=0)
ilr_std = ilr_samples.std(axis=0)
chi_mc_std = chi_samples.std(axis=0)
rho_mc_std = rho_samples.std(axis=0)

print(f"ILR predicted:   {ilr_mean.min():.2f} – {ilr_mean.max():.2f}")
print(f"ILR uncertainty: {ilr_std.min():.3f} – {ilr_std.max():.3f}")
print(f"χ uncertainty:   {chi_mc_std.min():.6f} – {chi_mc_std.max():.6f}")
print(f"Δρ uncertainty:  {rho_mc_std.min():.5f} – {rho_mc_std.max():.5f}")

# %%
comp_pred = ilr_inverse(ilr_mean)
Cu_pred = comp_pred[:, 0]

print(f"Cu predicted: {Cu_pred.min()*100:.2f}% – {Cu_pred.max()*100:.2f}%")
print(f"Cu true:      {Cu.min()*100:.2f}% – {Cu.max()*100:.2f}%")

# %% [markdown]
# ## Validation Metrics

# %%
encoder.eval()
dec_susc.eval()
dec_dens.eval()
dec_ilr.eval()

def r2_score(true, pred):
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - true.mean())**2)
    return 1 - ss_res / ss_tot

# Physical property metrics
chi_r2 = r2_score(susceptibility_true, chi_pred_np)
rho_r2 = r2_score(density_true, rho_pred_np)
cu_r2 = r2_score(Cu, Cu_pred)
cu_corr = np.corrcoef(Cu, Cu_pred)[0, 1]

# Validation ILR
with torch.no_grad():
    mu, _ = encoder(mag_grid_t, grav_grid_t, hyper_t, geochem_t)
    z_val = mu.unsqueeze(0).expand(val_coords_t.shape[0], -1)
    chi_val = chi_pred[val_cell_idx]
    rho_val = rho_pred[val_cell_idx]
    val_phi_t = all_phi_t[val_cell_idx]
    val_unit_ids_t = all_unit_ids_t[val_cell_idx]
    val_ilr_pred = dec_ilr(z_val, val_coords_t, val_phi_t, val_unit_ids_t, chi_val, rho_val)

val_rmse = float(torch.sqrt(torch.mean((val_ilr_pred - val_ilr_t)**2)))

print(f"--- Validation Metrics ---")
print(f"χ global   R²: {chi_r2:.4f}")
print(f"Δρ global  R²: {rho_r2:.4f}")
print(f"Cu global  R²: {cu_r2:.4f} | corr: {cu_corr:.4f}")
print(f"ILR Val  RMSE: {val_rmse:.4f}")

# %% [markdown]
# ---
# # Part 8: Visualizations

# %%
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
    'uncertainty_ilr': ilr_std[idx_slice, 0],
    'uncertainty_chi': chi_mc_std[idx_slice],
    'uncertainty_rho': rho_mc_std[idx_slice],
    'gempy_phi': scalar_field_norm[idx_slice],
    'lithology': cell_lith_id[idx_slice],
})

df_holes_train = df_train.drop_duplicates('hole')[['x', 'y']]
df_holes_val = df_val.drop_duplicates('hole')[['x', 'y']]

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='gempy_phi'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0.5)
    + lp.labs(title=f'GemPy scalar field φ (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

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
    + lp.geom_tile(mapping=lp.aes('x', 'y', fill='uncertainty_chi'))
    + lp.geom_point(data=df_holes_train, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.geom_point(data=df_holes_val, mapping=lp.aes('x', 'y'), shape=17, size=5, color='red')
    + lp.scale_fill_gradient(low='white', high='darkblue', name='σ_χ')
    + lp.labs(title=f'MC Dropout uncertainty — susceptibility (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('chi_true', 'chi_pred'))
    + lp.geom_point(alpha=0.5)
    + lp.geom_abline(slope=1, intercept=0, color='red', linetype='dashed')
    + lp.labs(title='Susceptibility: predicted vs true', x='χ true', y='χ predicted')
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
# ## GemPy + NeRF Decoders for Mineral Exploration
#
# This exercise demonstrates an architecture that bridges
# geological modeling (GemPy) with data-driven inversion (NeRF decoders),
# inspired by KoBold Metals' multi-sensor approach.
#
# ### Architecture Components
#
# | Component              | Tool / Method            | Role                           |
# |------------------------|--------------------------|--------------------------------|
# | Geological structure   | GemPy (implicit surface) | Scalar field φ + unit IDs      |
# | Multi-sensor fusion    | CNN encoder (PyTorch)    | 4 sensors → latent z           |
# | Property prediction    | NeRF decoders + φ input  | (z, coord, φ, unit) → χ, ρ    |
# | Composition prediction | ILR decoder              | (z, coord, φ, unit, χ, ρ) → Cu, Ni, Co |
# | Physics consistency    | SimPEG sensitivity       | G_mag @ χ, G_grav @ ρ         |
# | Uncertainty            | MC Dropout (50 draws)    | Per-cell epistemic uncertainty |
#
# ### Key Design Choices
#
# 1. **GemPy scalar field as decoder input**: The scalar field φ(x,y,z)
#    tells each decoder WHERE in the geological structure it's predicting.
#    This provides geological context that the neural network alone cannot
#    learn from surface observations. Unit embeddings further condition
#    predictions on rock type.
#
# 2. **Surface sensors as encoder-only inputs**: No decoder losses on
#    hyperspectral or soil geochemistry — avoids parasitic autoencoder
#    shortcuts that bypass the subsurface.
#
# 3. **Physical property bridge**: The ILR decoder receives χ and Δρ as
#    inputs, creating a direct differentiable pathway from geophysical
#    properties to mineral compositions.
#
# 4. **Borehole lithological logs**: In real exploration, geologists log
#    drill core lithology. These categorical labels feed into GemPy for
#    structural modeling and into the decoders via unit embeddings.
#
# ### Comparison to KoBold Metals' Approach
#
# | KoBold                          | This Exercise                        |
# |---------------------------------|--------------------------------------|
# | TerraShed (data platform)       | Multi-branch CNN encoder             |
# | Machine Prospector (ML engine)  | NeRF decoders + GemPy structure      |
# | Full-physics joint inversion    | G_mag @ χ + G_grav @ Δρ losses       |
# | Geological modeling             | GemPy implicit surfaces              |
# | Drill core logging              | Lithological logs → unit embeddings  |
# | Uncertainty quantification      | MC Dropout (50 draws)                |
#
# ### Potential Extension: Pyro Integration
#
# Since GemPy (PyTorch) and the decoders (PyTorch) share the same autodiff
# framework, the entire pipeline could be wrapped in a Pyro probabilistic
# model for full Bayesian inference over both geological structure (contact
# positions, orientations) and physical properties (decoder weights)
# simultaneously. This would provide joint structural-property uncertainty.
#
# ### References
#
# - KoBold Metals. AI-powered mineral exploration.
#   IEEE Spectrum (2024): https://spectrum.ieee.org/ai-mining
# - de la Varga et al. (2019). GemPy 1.0: open-source stochastic
#   geological modeling and inversion. Geosci. Model Dev.
# - Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
# - Eagar, K. (SimPEG). Open-source geophysical modeling.
