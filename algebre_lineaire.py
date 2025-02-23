#partie1
import numpy as np
import matplotlib.pyplot as plt

def euler_method(r, K, P0, dt, t_max):
    """
    Résout l'équation logistique par la méthode d'Euler.

    Args:
        r: Taux de croissance intrinsèque.
        K: Capacité limite.
        P0: Population initiale.
        dt: Pas de temps.
        t_max: Temps maximal de simulation.

    Returns:
        Un tuple (temps, populations) contenant les valeurs de temps et les
        populations correspondantes calculées à chaque pas de temps.
    """
    temps = np.arange(0, t_max, dt)  # Crée un tableau de valeurs de temps
    populations = [P0]  # Initialise la liste des populations avec P0

    for i in range(len(temps) - 1):
        P_current = populations[-1]  # Dernière valeur de population calculée
        dPdt = r * P_current * (1 - P_current / K)  # Calcule dP/dt
        P_next = P_current + dt * dPdt  # Calcule la prochaine population
        populations.append(P_next)  # Ajoute la nouvelle population à la liste

    return temps, populations

# Paramètres
r = 0.05
K = 100000
P0 = 1000
dt = 0.1  # Choisis un pas de temps. Plus il est petit, plus c'est précis (mais plus long à calculer)
t_max = 200  # Simule sur 200 unités de temps (par exemple, des mois)

# Résolution
temps, populations_euler = euler_method(r, K, P0, dt, t_max)

# Affichage des résultats (courbe)
plt.plot(temps, populations_euler)
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Croissance logistique (Méthode d'Euler)")
plt.grid(True)
plt.show()

# Pour trouver quand la population atteint 50000:
for i, P in enumerate(populations_euler):
    if P >= 50000:
        print(f"La population atteint 50000 au temps t = {temps[i]:.2f}")
        break

#partie 2
def solution_analytique(r, K, P0, t):
    """
    Calcule la solution analytique de l'équation logistique.

    Args:
        r: Taux de croissance.
        K: Capacité limite.
        P0: Population initiale.
        t: Temps (peut être un tableau NumPy).

    Returns:
        La population P(t) au temps t.
    """
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

# Calcul des populations avec la solution analytique
populations_analytique = solution_analytique(r, K, P0, temps)  # On utilise le même tableau 'temps'
from sklearn.metrics import mean_squared_error

# Calcul de la MSE
mse = mean_squared_error(populations_analytique, populations_euler)
print(f"MSE entre la méthode d'Euler et la solution analytique : {mse:.2f}")

# MSE sur des intervalles de temps spécifiques (par exemple, les 50 premiers mois, puis les 50 suivants, etc.)
intervalle = 50
for i in range(0, len(temps), intervalle):
  debut = i
  fin = min(i + intervalle, len(temps)) # pour ne pas dépasser la fin du tableau
  temps_partiel = temps[debut:fin]
  pop_euler_partiel = populations_euler[debut:fin]
  pop_analytique_partiel = populations_analytique[debut:fin]

  mse_partiel = mean_squared_error(pop_analytique_partiel, pop_euler_partiel)
  print(f"MSE sur l'intervalle [{temps_partiel[0]:.1f}, {temps_partiel[-1]:.1f}]: {mse_partiel:.2f}")
plt.plot(temps, populations_euler, label="Méthode d'Euler")
plt.plot(temps, populations_analytique, label="Solution analytique")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Comparaison Euler vs. Solution Analytique")
plt.legend()
plt.grid(True)
plt.show()

#partie 3

# Charger les données 
import pandas as pd
data = pd.read_csv("Dataset_nombre_utilisateurs.csv")

temps_reel = data["Jour"]
utilisateurs_reel = data["Utilisateurs"]

plt.plot(temps_reel, utilisateurs_reel, label="Données réelles")
plt.xlabel("Jour")
plt.ylabel("Nombre d'utilisateurs")
plt.title("Nombre d'utilisateurs réels")
plt.legend()
plt.grid(True)
plt.show()
# Estimation de K à partir des données (par exemple, la valeur maximale observée)
K_estime = utilisateurs_reel.max()

# 50% de saturation
saturation_50 = K_estime / 2
jour_saturation_50 = temps_reel[utilisateurs_reel >= saturation_50].iloc[0]  # Premier jour où on dépasse 50%
print(f"50% de saturation atteint le jour {jour_saturation_50}")

# Saturation (exemple avec un seuil de 95%)
seuil_saturation = 0.95 * K_estime
jour_saturation = temps_reel[utilisateurs_reel >= seuil_saturation].iloc[0]
print(f"Saturation (95%) atteinte le jour {jour_saturation}")

# Interpolation de la méthode d'Euler
populations_euler_interp = np.interp(temps_reel, temps, populations_euler)

# Interpolation de la solution analytique
populations_analytique_interp = np.interp(temps_reel, temps, populations_analytique)

# Calcul des MSE
mse_euler_reel = mean_squared_error(utilisateurs_reel, populations_euler_interp)
mse_analytique_reel = mean_squared_error(utilisateurs_reel, populations_analytique_interp)

print(f"MSE (Euler vs. Réel) : {mse_euler_reel:.2f}")
print(f"MSE (Analytique vs. Réel) : {mse_analytique_reel:.2f}")

# MSE par intervalles 
intervalle = 50 #jours
for i in range(0, len(temps_reel), intervalle):
    debut = i
    fin = min(i+intervalle, len(temps_reel))

    temps_partiel = temps_reel[debut:fin]
    utilisateurs_reel_partiel = utilisateurs_reel[debut:fin]
    pop_euler_partiel = populations_euler_interp[debut:fin]
    pop_analytique_partiel = populations_analytique_interp[debut:fin]

    mse_euler_partiel = mean_squared_error(utilisateurs_reel_partiel, pop_euler_partiel)
    mse_analytique_partiel = mean_squared_error(utilisateurs_reel_partiel, pop_analytique_partiel)

    print(f"MSE sur l'intervalle [{temps_partiel.iloc[0]:.1f}, {temps_partiel.iloc[-1]:.1f}]:")
    print(f"  Euler: {mse_euler_partiel:.2f}")
    print(f"  Analytique: {mse_analytique_partiel:.2f}")
    plt.plot(temps_reel, utilisateurs_reel, label="Données réelles")
    plt.plot(temps_reel, populations_euler_interp, label="Méthode d'Euler (interpolée)")
    plt.plot(temps_reel, populations_analytique_interp, label="Solution analytique (interpolée)")
    plt.xlabel("Jour")
    plt.ylabel("Nombre d'utilisateurs")
    plt.title("Comparaison Modèles vs. Réel")
    plt.legend()
    plt.grid(True)
    plt.show()
    from scipy.optimize import curve_fit

def modele_logistique_pour_fit(t, r, K, P0):
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

# Ajustement des paramètres
parametres_optimaux, covariance = curve_fit(
    modele_logistique_pour_fit,
    temps_reel,
    utilisateurs_reel,
    p0=[0.05, 100000, 1000],  # Valeurs initiales pour r, K, P0
    bounds=([0,0,0],[1,1000000,10000]) #limites pour éviter des valeurs aberrantes
)

r_optimal, K_optimal, P0_optimal = parametres_optimaux

print(f"Paramètres optimaux : r={r_optimal:.4f}, K={K_optimal:.0f}, P0={P0_optimal:.0f}")

# Calcul des prédictions avec les paramètres optimaux
populations_optimales = modele_logistique_pour_fit(temps_reel, r_optimal, K_optimal, P0_optimal)

# Nouvelle MSE
mse_optimale = mean_squared_error(utilisateurs_reel, populations_optimales)
print(f"MSE après recalibrage : {mse_optimale:.2f}")

# Tracé
plt.plot(temps_reel, utilisateurs_reel, label="Données réelles")
plt.plot(temps_reel, populations_optimales, label="Modèle recalibré")
plt.xlabel("Jour")
plt.ylabel("Nombre d'utilisateurs")
plt.title("Modèle recalibré sur les données réelles")
plt.legend()
plt.grid(True)
plt.show()
# Partie 4: Question ouverte

# Hypothèses pour le nouveau produit
max_users_per_server = 2000
server_cost = 1000  # par mois
user_acquisition_cost = 10
marketing_budget = 50000
subscription_fee = 11.99
market_size = 400000
market_share = 0.2
initial_users = 500

def project_revenue_and_costs(months, r, initial_users, market_size, market_share, subscription_fee, max_users_per_server, server_cost, user_acquisition_cost, marketing_budget):
    expected_users = initial_users * (1 + r) ** months
    expected_users = np.minimum(expected_users, market_size * market_share)
    revenue = expected_users * subscription_fee
    servers_needed = np.ceil(expected_users / max_users_per_server)
    cost_servers = servers_needed * server_cost
    cost_acquisition = expected_users * user_acquisition_cost
    total_cost = cost_servers + cost_acquisition

    # Répartition du budget marketing
    marketing_spend = np.zeros_like(months, dtype=float)
    marketing_spend[0] = marketing_budget * 0.35
    remaining_budget = marketing_budget * 0.65
    marketing_spend[1:] = remaining_budget / (len(months) - 1)
    total_cost += marketing_spend

    profit = revenue - total_cost
    return revenue, total_cost, profit

# Projection sur 12 mois
months = np.arange(1, 13)
revenue, total_cost, profit = project_revenue_and_costs(
    months, r, initial_users, market_size, market_share,
    subscription_fee, max_users_per_server, server_cost,
    user_acquisition_cost, marketing_budget
)

# Tracer le plan de projection
plt.figure(figsize=(10, 6))
plt.plot(months, revenue, label='Revenu')
plt.plot(months, total_cost, label='Coûts totaux')
plt.plot(months, profit, label='Bénéfice', linestyle='--')
plt.xlabel('Mois')
plt.ylabel('Euros')
plt.title('Projection du chiffre d\'affaires et des bénéfices')
plt.legend()
plt.grid()
plt.show()

# Afficher la rentabilité
for i, p in enumerate(profit):
    status = "Rentable" if p > 0 else "Déficit"
    print(f"Mois {i+1}: Bénéfice = {p:.2f} EUR ({status})")
