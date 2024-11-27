import pandas as pd
import numpy as np
from hidden_markov_model import HiddenMarkovModel

# Lees de data in
csv_file = "alledatahmm.csv"  # Zorg dat dit pad klopt
df = pd.read_csv(csv_file, sep=",")  # Zorg ervoor dat je de juiste scheidingsteken gebruikt

# Lijst van toegestane kleuren
toegestane_kleuren = ["blauw", "rood", "geel", "groen"]

# Controleer welke waarden in de 'Kleur' kolom niet in de lijst van toegestane kleuren staan
ongeldige_kleuren = df[~df['Kleur'].isin(toegestane_kleuren)]

# Print de rijen met ongeldige kleuren (optioneel)
if not ongeldige_kleuren.empty:
    print("Rijen met ongeldige kleuren:")
    print(ongeldige_kleuren)

# Optie 1: Vervang ongeldige kleuren met 'onbekend'
df['Kleur'] = df['Kleur'].apply(lambda x: x if x in toegestane_kleuren else 'onbekend')

# Optie 2: Verwijder rijen met ongeldige kleuren
# df = df[df['Kleur'].isin(toegestane_kleuren)]

# Kleur omzetten naar integers
kleur_mapping = {"blauw": 0, "geel": 1, "groen": 2, "rood": 3, "onbekend": -1}
df['Kleur'] = df['Kleur'].map(kleur_mapping).astype(int)  # Forceer naar integers

# Extract kolommen
emissions = df['Kleur'].tolist()
states = (df['Tafel'] - 1).tolist()  # Zet tafels om naar 0-index

# Hidden Markov Model instellen
startprob = np.array([1/3, 1/3, 1/3])
transmat = np.array([
    [1/3, 1/3, 1/3],
    [1/6, 1/3, 1/2],
    [1/2, 1/6, 1/3],
])
emissionprob = np.array([
    [1/2, 1/4, 1/12, 1/6],  
    [1/6, 1/2, 1/6, 1/6],   
    [1/12, 0, 1/2, 5/12],    
])
emissionprob = emissionprob + 1e-6  # Voeg pseudotellen toe
emissionprob = emissionprob / emissionprob.sum(axis=1, keepdims=True)  # Normaliseer opnieuw

# Model initialiseren
model = HiddenMarkovModel(n_components=3, n_features=4)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

# Pak de eerste 5 waarnemingen
first_five_emissions = emissions[:5]
first_five_states = states[:5]

# Debugging: Print de waarden
print("Eerste 5 emissions:", first_five_emissions)
print("Eerste 5 states:", first_five_states)

# Log-likelihood berekeningen
log_prob_first_five = model.score(first_five_emissions, first_five_states)
print(f"Log-waarschijnlijkheid eerste 5 waarnemingen: {log_prob_first_five:.3f}")

# Reken volledige log-waarschijnlijkheid uit
log_prob_full = model.score(emissions, states)
print(f"Log-waarschijnlijkheid volledige reeks: {log_prob_full:.3f}")
