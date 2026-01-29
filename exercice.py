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
#           │ latent → χ(3D) │                  │ latent+(x,y,z) │               │
#           └────────────────┘                  │    → ILR       │               │
#                    │                          └────────────────┘               │
#                    ▼                                   │                       │
#           ┌────────────────┐                          │                       │
#           │ FORWARD MODEL  │                          ▼                       │
#           │  (physique)    │                   Loss reconstruction            │
#           └────────────────┘                   (vs forages)                   │
#                    │                                                          │
#                    ▼                                                          │
#             Loss physique ◀───────────────────────────────────────────────────┘
#             (vs magnétique observé)
# ```
#
# Le PINN = la loss physique force la cohérence avec les observations.

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
def ilr_transform(comp):
    """Compositions (n, 4) → ILR (n, 3)"""
    log_comp = np.log(comp + 1e-10)
    # Matrice de contraste pour ILR (Helmert)
    # Pour 4 composantes, on a une matrice 3x4
    V = np.array([
        [np.sqrt(3/4), -1/np.sqrt(12), -1/np.sqrt(12), -1/np.sqrt(12)],
        [0, np.sqrt(2/3), -1/np.sqrt(6), -1/np.sqrt(6)],
        [0, 0, np.sqrt(1/2), -np.sqrt(1/2)]
    ])
    return log_comp @ V.T

def ilr_inverse(ilr):
    """ILR (n, 3) → Compositions (n, 4)"""
    # Inverse: V.T @ V = I, donc on utilise V.T pour reconstruire
    V = np.array([
        [np.sqrt(3/4), -1/np.sqrt(12), -1/np.sqrt(12), -1/np.sqrt(12)],
        [0, np.sqrt(2/3), -1/np.sqrt(6), -1/np.sqrt(6)],
        [0, 0, np.sqrt(1/2), -np.sqrt(1/2)]
    ])
    log_comp = ilr @ V
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
    amplitude=55000, # |B| en nT # en pratique le champ est variable, mais simpeg sais pas trop quoi faire avec la réalité d'un champ variable
    inclination=70, # angle vertical
    declination=0 # angle horizontal
)
survey = magnetics.Survey(source)

# Simulation
sim = magnetics.Simulation3DIntegral(
    mesh=mesh, # La grille 3D de cellules
    survey=survey, # Où mesurer + champ de fond
    chiMap=maps.IdentityMap(mesh),# χ_input → χ_model (ici: pas de transformation)
    active_cells=np.ones(n_cells, dtype=bool),# Quelles cellules sont "actives" (pas de l'air)
    store_sensitivities='forward_only'# Ne pas stocker la matrice jacobienne (économie mémoire)
)
mag_observed = sim.dpred(susceptibility_true) + np.random.randn(len(stations)) * 1.5

print(f"Magnétique: {mag_observed.min():.1f} – {mag_observed.max():.1f} nT")

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
# Plus de forages pour mieux contraindre
hole_xy = [
    (1500, 1500), (2800, 2500), (500, 500), (2000, 2000), (3500, 1000),
    (1200, 1800), (2500, 2200), (1800, 2800), (3000, 1500), (800, 2500)
]
forages = []
for i, (hx, hy) in enumerate(hole_xy):
    ix, iy = np.argmin(np.abs(mesh.cell_centers_x - hx)), np.argmin(np.abs(mesh.cell_centers_y - hy))
    for iz in range(nz):
        idx = ix + iy * nx + iz * nx * ny
        forages.append({
            'hole': f'DH{i+1}', 'x': mesh.cell_centers_x[ix], 'y': mesh.cell_centers_y[iy],
            'z': mesh.cell_centers_z[iz], 'Cu': Cu[idx], 'Ni': Ni[idx], 
            'ilr0': ilr_true[idx, 0], 'ilr1': ilr_true[idx, 1], 'ilr2': ilr_true[idx, 2]
        })
df_forages = pd.DataFrame(forages)
print(f"Échantillons: {len(df_forages)}")

# %% [markdown]
# ---
# # Partie 2: Forward model différentiable en JAX
#
# SimPEG n'est pas différentiable. On écrit une version simplifiée en JAX
# pour pouvoir backpropager le gradient.

# %%
@jit
def forward_magnetic_jax(susceptibility, stations, cell_centers, cell_volume, B0=55000.0):
    """
    Forward magnétique simplifié (approximation dipôle).
    
    χ(x) contribue au champ mesuré en s avec une atténuation 1/r³
    """
    def contribution_at_station(station):
        # Vecteur station - cellule
        r_vec = station - cell_centers  # (n_cells, 3)
        r = jnp.sqrt(jnp.sum(r_vec**2, axis=1) + 1e-6)  # (n_cells,)
        
        # Composante verticale simplifiée (inclinaison ~70°)
        cos_theta = r_vec[:, 2] / r
        
        # Contribution dipôle
        contrib = susceptibility * B0 * cell_volume * (3 * cos_theta**2 - 1) / (4 * jnp.pi * r**3)
        return jnp.sum(contrib)
    
    return vmap(contribution_at_station)(stations)

# Test
cc_jax = jnp.array(cc)
stations_jax = jnp.array(stations)
cell_volume = dx * dy * dz

mag_test = forward_magnetic_jax(jnp.array(susceptibility_true), stations_jax, cc_jax, cell_volume)
print(f"Forward JAX: {float(mag_test.min()):.1f} – {float(mag_test.max()):.1f}")

# %% [markdown]
# ---
# # Partie 3: Architecture du réseau

# %%
class Encoder(nn.Module):
    """Magnétique 2D → distribution sur le latent"""
    latent_dim: int = 32
    
    @nn.compact
    def __call__(self, mag_grid):
        x = mag_grid.ravel()
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)
        return mu, log_var


class DecoderSusceptibility(nn.Module):
    """Latent → susceptibilité 3D (une valeur par cellule)"""
    n_cells: int
    
    @nn.compact
    def __call__(self, latent):
        x = nn.Dense(64)(latent)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        # Sortie: log-susceptibilité (pour garantir χ > 0)
        log_chi = nn.Dense(self.n_cells)(x)
        return jnp.exp(log_chi) * 0.01  # scaling pour être dans le bon range


class DecoderILR(nn.Module):
    """Latent + coordonnées normalisées → ILR à cette position"""
    
    @nn.compact
    def __call__(self, latent, coord_norm):
        x = jnp.concatenate([latent, coord_norm])
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        ilr = nn.Dense(3)(x)
        return ilr

# %% [markdown]
# ---
# # Partie 4: Loss function avec PINN

# %%
def loss_fn(params, encoder, dec_susc, dec_ilr, 
            mag_grid, mag_observed,
            train_coords_norm, train_ilr,
            stations, cell_centers, cell_volume,
            rng_key, lambda_physics=1.0, lambda_kl=0.01):
    """
    Loss totale = reconstruction ILR + KL + physique
    
    La loss physique (PINN) force la susceptibilité décodée à reproduire
    les observations magnétiques quand on la passe dans le forward model.
    """
    
    # --- Encoder ---
    mu, log_var = encoder.apply(params['encoder'], mag_grid)
    
    # Reparameterization trick
    std = jnp.exp(0.5 * log_var)
    eps = random.normal(rng_key, mu.shape)
    z = mu + std * eps
    
    # --- Decoder susceptibilité ---
    chi_pred = dec_susc.apply(params['dec_susc'], z)
    
    # --- Forward model (PINN) ---
    mag_pred = forward_magnetic_jax(chi_pred, stations, cell_centers, cell_volume)
    
    # Loss physique: le magnétique prédit doit matcher l'observé
    loss_physics = jnp.mean((mag_pred - mag_observed)**2) / jnp.var(mag_observed)
    
    # --- Decoder ILR ---
    def predict_ilr(coord):
        return dec_ilr.apply(params['dec_ilr'], z, coord)
    
    ilr_pred = vmap(predict_ilr)(train_coords_norm)
    
    # Loss reconstruction ILR
    loss_recon = jnp.mean((ilr_pred - train_ilr)**2)
    
    # --- KL divergence ---
    loss_kl = -0.5 * jnp.mean(1 + log_var - mu**2 - jnp.exp(log_var))
    
    # --- Total ---
    loss_total = loss_recon + lambda_kl * loss_kl + lambda_physics * loss_physics
    
    return loss_total, {
        'total': loss_total, 'recon': loss_recon, 
        'kl': loss_kl, 'physics': loss_physics,
        'chi_mean': jnp.mean(chi_pred), 'chi_max': jnp.max(chi_pred)
    }

# %% [markdown]
# ---
# # Partie 5: Entraînement

# %%
# Initialiser
rng = random.PRNGKey(42)
rng, *init_rngs = random.split(rng, 4)

latent_dim = 32
encoder = Encoder(latent_dim=latent_dim)
dec_susc = DecoderSusceptibility(n_cells=n_cells)
dec_ilr = DecoderILR()

# Dummy inputs
dummy_mag = jnp.zeros((n_stations, n_stations))
dummy_latent = jnp.zeros(latent_dim)
dummy_coord = jnp.zeros(3)

params = {
    'encoder': encoder.init(init_rngs[0], dummy_mag),
    'dec_susc': dec_susc.init(init_rngs[1], dummy_latent),
    'dec_ilr': dec_ilr.init(init_rngs[2], dummy_latent, dummy_coord)
}

# %%
# Préparer les données
mag_grid = jnp.array(mag_observed.reshape(n_stations, n_stations))
mag_grid_norm = (mag_grid - mag_grid.mean()) / (mag_grid.std() + 1e-6)

mag_obs_jax = jnp.array(mag_observed)
stations_jax = jnp.array(stations)
cc_jax = jnp.array(cc)

# Coordonnées des forages normalisées
coord_mean = jnp.array([nx*dx/2, ny*dy/2, -nz*dz/2])
coord_std = jnp.array([nx*dx/2, ny*dy/2, nz*dz/2])
train_coords_norm = jnp.array((df_forages[['x', 'y', 'z']].values - np.array(coord_mean)) / np.array(coord_std))
train_ilr = jnp.array(df_forages[['ilr0', 'ilr1', 'ilr2']].values)

print(f"Prêt pour l'entraînement")

# %%
# Optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jit
def train_step(params, opt_state, rng_key):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, encoder, dec_susc, dec_ilr,
        mag_grid_norm, mag_obs_jax,
        train_coords_norm, train_ilr,
        stations_jax, cc_jax, cell_volume,
        rng_key, lambda_physics=0.05, lambda_kl=0.001
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# %%
# Boucle d'entraînement
history = []
n_epochs = 2000

for epoch in range(n_epochs):
    rng, step_rng = random.split(rng)
    params, opt_state, loss, aux = train_step(params, opt_state, step_rng)
    history.append({**{k: float(v) for k, v in aux.items()}, 'epoch': epoch})
    
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | loss={aux['total']:.4f} | recon={aux['recon']:.4f} | "
              f"physics={aux['physics']:.4f} | χ_max={aux['chi_max']:.4f}")

# %%
# Courbes de loss
df_hist = pd.DataFrame(history)

(
    lp.ggplot(df_hist, lp.aes(x='epoch'))
    + lp.geom_line(lp.aes(y='recon'), color="#962561")
    + lp.geom_line(lp.aes(y='physics'), color="#abc000")
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
mag_from_pred = forward_magnetic_jax(chi_pred, stations_jax, cc_jax, cell_volume)
print(f"Mag prédit: {float(mag_from_pred.min()):.1f} – {float(mag_from_pred.max()):.1f}")
print(f"Mag observé: {mag_observed.min():.1f} – {mag_observed.max():.1f}")

# %% [markdown]
# ## Prédire les ILR avec incertitude (sampling)

# %%
# Grille de prédiction: toutes les cellules
all_coords_norm = (cc - np.array(coord_mean)) / np.array(coord_std)
all_coords_norm_jax = jnp.array(all_coords_norm)

# Sampler plusieurs latents
n_samples = 30
ilr_samples = []

for i in range(n_samples):
    rng, sample_rng = random.split(rng)
    std = jnp.exp(0.5 * log_var)
    eps = random.normal(sample_rng, mu.shape)
    z = mu + std * eps
    
    def predict_ilr(coord):
        return dec_ilr.apply(params['dec_ilr'], z, coord)
    
    ilr_sample = vmap(predict_ilr)(all_coords_norm_jax)
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

df_holes = df_forages.drop_duplicates('hole')[['x', 'y']]

# %%
(
    lp.ggplot(df_results)
    + lp.geom_tile(mapping= lp.aes('x', 'y', fill='chi_pred'))
    + lp.geom_point(data=df_holes, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.scale_fill_gradient(low='white', high='darkblue', name='χ (SI)')
    + lp.labs(title=f'Susceptibilité prédite (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('x', 'y'))
    + lp.geom_tile(mapping=lp.aes(fill='chi_true'))
    + lp.geom_point(data=df_holes, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.scale_fill_gradient(low='white', high='darkblue', name='χ (SI)')
    + lp.labs(title=f'Susceptibilité vraie (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('x', 'y'))
    + lp.geom_tile(mapping=lp.aes(fill='Cu_pred'))
    + lp.geom_point(data=df_holes, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.scale_fill_gradient(low='white', high='darkred', name='Cu %')
    + lp.labs(title=f'Cu prédit (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('x', 'y', fill='Cu_true'))
    + lp.geom_tile()
    + lp.geom_point(data=df_holes, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
    + lp.scale_fill_gradient(low='white', high='darkred', name='Cu %')
    + lp.labs(title=f'Cu vrai (z={mesh.cell_centers_z[iz]:.0f}m)', x='X', y='Y')
)

# %%
(
    lp.ggplot(df_results, lp.aes('x', 'y'))
    + lp.geom_tile(mapping=lp.aes(fill='uncertainty'))
    + lp.geom_point(data=df_holes, mapping=lp.aes('x', 'y'), shape=17, size=5, color='black')
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
#    - Encoder: magnétique → latent
#    - Decoder susceptibilité: latent → χ 3D
#    - Decoder ILR: latent + (x,y,z) → compositions
#    - Forward JAX: χ → magnétique prédit
#    - Loss physique: ||mag_prédit - mag_observé||²
# 4. **Incertitude** par sampling du latent
#
# ## Ce qui marche
#
# - La contrainte physique (PINN) force la susceptibilité à être cohérente
# - L'architecture apprend à combiner les deux sources d'information
# - L'incertitude reflète (partiellement) notre ignorance
#
# ## Limites
#
# - Forward simplifié (pas exact comme SimPEG)
# - Peu de données → généralisation limitée
# - Pas de multi-physique (EM, gravité)
#
# ## Pour Marco
#
# "J'ai implémenté un encodeur-décodeur avec régularisation physique pour 
# l'inversion magnétique. Le PINN force la cohérence entre la géologie 
# inférée et les observations de surface. L'incertitude vient du sampling 
# du latent. Je n'ai pas utilisé de GP car le problème est une inversion 
# contrainte par la physique, pas une interpolation spatiale."
