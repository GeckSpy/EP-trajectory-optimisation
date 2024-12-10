import subprocess
import time

def f(x):
    n = 0
    for i in range(x):
        n += i
    return n

x = 100000000

# Lancer powermetrics en arrière-plan
powermetrics_process = subprocess.Popen(
    ["sudo", "powermetrics", "--samplers", "cpu_power", "-i", "500"],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True
)

# Attendre un court instant pour initialiser powermetrics
time.sleep(1)

# Exécuter la fonction
start_time = time.time()
result = f(x)
end_time = time.time()

# Arrêter powermetrics
powermetrics_process.terminate()

# Lire et traiter les données de powermetrics
output, _ = powermetrics_process.communicate()
cpu_power_lines = [
    float(line.split(":")[1].strip().split()[0])
    for line in output.splitlines() if "CPU power" in line
]

# Calculer l'énergie totale consommée
interval = 0.5  # 500 ms
total_energy = sum(cpu_power_lines) * interval  # en joules

print(f"Résultat de f({x}) : {result}")
print(f"Temps d'exécution : {end_time - start_time} secondes")
print(f"Énergie consommée : {total_energy} joules")
