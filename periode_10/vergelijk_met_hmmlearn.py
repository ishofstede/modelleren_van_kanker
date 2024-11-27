import numpy as np
from hmmlearn import hmm
from hidden_markov_model import HiddenMarkovModel

# Parameters
startprob = np.array([1 / 3, 1 / 3, 1 / 3])
transmat = np.array([
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 6, 1 / 3, 1 / 2],
    [1 / 2, 1 / 6, 1 / 3],
])
emissionprob = np.array([
    [1 / 2, 1 / 4, 1 / 12, 1 / 6],
    [1 / 6, 1 / 2, 1 / 6, 1 / 6],
    [1 / 12, 0, 1 / 2, 5 / 12],
])

# Emissions and states
emissions = [1, 2, 1, 2, 0]  # indices van de emissies
states = [1, 2, 1, 2, 0]

# Eigen implementatie
eigen_model = HiddenMarkovModel(n_components=3, n_features=4)
eigen_model.startprob_ = startprob
eigen_model.transmat_ = transmat
eigen_model.emissionprob_ = emissionprob

log_prob_eigen = eigen_model.score(emissions, states)
print(f"Eigen module:   ln(p) = {log_prob_eigen:.3f}")

# Converteer emissies naar one-hot representatie
n_features = emissionprob.shape[1]
emissions_one_hot = np.zeros((len(emissions), n_features), dtype=int)
for i, e in enumerate(emissions):
    emissions_one_hot[i, e] = 1

# hmmlearn implementatie
hmmlearn_model = hmm.MultinomialHMM(n_components=3)
hmmlearn_model.startprob_ = startprob
hmmlearn_model.transmat_ = transmat
hmmlearn_model.emissionprob_ = emissionprob
hmmlearn_model.n_trials = 1  # Zet aantal trials

# Gebruik de one-hot representatie
log_prob_hmmlearn = hmmlearn_model.score(emissions_one_hot)
print(f"hmmlearn:       ln(p) = {log_prob_hmmlearn:.3f}")

# Controleer of ze overeenkomen
if np.isclose(log_prob_eigen, log_prob_hmmlearn):
    print("Het resultaat van de eigen module komt overeen met dat van hmmlearn!")
else:
    print("De resultaten komen niet overeen. Controleer je implementatie.")
