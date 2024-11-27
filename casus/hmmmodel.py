import numpy as np
from itertools import product
from math import log, exp
from hmmlearn.hmm import CategoricalHMM

class HiddenMarkovModel:
    def __init__(self, n_components, n_features):
        self.n_components = n_components
        self.n_features = n_features
        self.startprob_ = np.ones(n_components) / n_components  # Uniforme verdeling
        self.transmat_ = np.ones((n_components, n_components)) / n_components  # Uniforme overgang
        self.emissionprob_ = np.ones((n_components, n_features)) / n_features  # Uniforme emissie

    def score(self, emissions, state_sequence):
        log_prob = 0.0
        log_prob += log(self.startprob_[state_sequence[0]])  # start state
        for t in range(1, len(state_sequence)):
            prev_state = state_sequence[t-1]
            current_state = state_sequence[t]
            log_prob += log(self.transmat_[prev_state, current_state])  # transition
            log_prob += log(self.emissionprob_[current_state, emissions[t]])  # emission
        return log_prob

    def calculate_log_prob_all_states(self, emissions):
        prob_sum = 0.0
        for state_sequence in product(range(self.n_components), repeat=len(emissions)):  
            prob_sum += exp(self.score(emissions, state_sequence)) 
        log_prob = log(prob_sum)
        p_value = exp(log_prob)  # Bereken de kans p van de log-kans
        return log_prob, p_value


def hmmlearn_comparison(emissions):
    # Aantal toestanden (n_components) en emissies (n_features)
    n_components = 3
    n_features = 4

    # Initialisatie van een CategoricalHMM-model met dezelfde parameters
    model_hmmlearn = CategoricalHMM(n_components=n_components, n_iter=1000)

    # Set de transitie- en emissiematrices (gebruik uniforme waarden voor vergelijking)
    model_hmmlearn.startprob_ = np.ones(n_components) / n_components
    model_hmmlearn.transmat_ = np.ones((n_components, n_components)) / n_components
    model_hmmlearn.emissionprob_ = np.ones((n_components, n_features)) / n_features

    # Zet de emissies om in de juiste vorm voor hmmlearn
    emissions_hmmlearn = np.array(emissions).reshape(-1, 1)

    # Bereken de log-kans en de kans (score)
    log_prob_hmmlearn = model_hmmlearn.score(emissions_hmmlearn)
    p_value_hmmlearn = exp(log_prob_hmmlearn)  # Om de kans in plaats van log-kans te berekenen

    return log_prob_hmmlearn, p_value_hmmlearn

# Testdata
emissions = [1, 2, 0, 3, 2]

# Bereken de log-kans voor  model
model = HiddenMarkovModel(n_components=3, n_features=4)
log_prob, p_value = model.calculate_log_prob_all_states(emissions)

# Print de resultaten van module
print(f"Eigen module:   ln(p) = {log_prob:.3f}   (p = {p_value:.3e})")

# Bereken de log-kans voor hmmlearn
log_prob_hmmlearn, p_value_hmmlearn = hmmlearn_comparison(emissions)

# Print
print(f"hmmlearn:       ln(p) = {log_prob_hmmlearn:.3f}   (p = {p_value_hmmlearn:.3e})")
