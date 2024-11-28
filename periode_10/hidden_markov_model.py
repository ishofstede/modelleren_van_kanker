import numpy as np
from math import log

class HiddenMarkovModel:
    def __init__(self, n_components, n_features):
        
        self.n_components = n_components  # Aantal toestanden in dit geval 3 voor elke tafel 1 
        self.n_features = n_features    # Aantal mogelijke emissies in dit geval 4 voor elke kleur 
        self.startprob_ = np.zeros(n_components)  # start kans, in dit geval 3 voor elke tafel
        self.transmat_ = np.zeros((n_components, n_components))  # De kans om van een toestand naar een andere toestand te gaan 
        self.emissionprob_ = np.zeros((n_components, n_features))  # De kans om een kleur te krijgen bij een specifieke toestand 

    def __str__(self):
        # in string mode om het leesbaar te maken 
        return (f"HiddenMarkovModel(n_components={self.n_components}, "
                f"n_features={self.n_features})\n"
                f"Startprobabilities: {self.startprob_}\n"
                f"Transition matrix:\n{self.transmat_}\n"
                f"Emission matrix:\n{self.emissionprob_}")

    def __repr__(self):
        # geeft een samenvatting van het object dat leesbaar is.
        return (f"HiddenMarkovModel(n_components={self.n_components}, "
                f"n_features={self.n_features})")

    
    def sample(self, n_samples):
        # deze functie genereerd de sample data aan de hand van wat de kansen zijn voor de overgangsmatrix/kansen. 
        states = []
        emissions = []
        
        # de som moet altijd 1 zijn. 
        self.startprob_ = self.startprob_ / self.startprob_.sum()
        self.transmat_ = np.array([row / row.sum() if row.sum() != 0 else row for row in self.transmat_])
        self.emissionprob_ = np.array([row / row.sum() if row.sum() != 0 else row for row in self.emissionprob_])

        # het random kiezen van een toestand
        current_state = np.random.choice(self.n_components, p=self.startprob_)
        states.append(current_state)
        
        for _ in range(n_samples):
            
            # per stap wordt de emissie gegenereerd 
            emission = np.random.choice(self.n_features, p=self.emissionprob_[current_state])
            emissions.append(emission)
            
            # De current states met de random kansen 
            current_state = np.random.choice(self.n_components, p=self.transmat_[current_state])
            states.append(current_state)
        return emissions, states[:-1]  # Laatste toestand verwijderen om gelijke lengte te hebben
    
    def forward_algorithm(self, X):
        T = len(X)  # Length of the observation sequence
        fwd = np.zeros((T, self.n_components))
        
        # Initialize the forward matrix for the first time step
        fwd[0, :] = self.startprob_ * self.emissionprob_[:, X[0]]
        
        # Recursively fill in the forward matrix
        for t in range(1, T):
            for s in range(self.n_components):
                fwd[t, s] = np.sum(fwd[t-1, :] * self.transmat_[:, s]) * self.emissionprob_[s, X[t]]
        
        # Return the log-likelihood by summing over all states at the last time step
        log_prob = np.log(np.sum(fwd[T-1, :]))
        return log_prob
    
    def score(self, X, state_sequence = None):
        if state_sequence is None:
            log_prob = self.forward_algorithm(X)  # Use the forward algorithm
        else:
            # Implement from Part II: Calculate log-likelihood of emissions and states
            if not all(0 <= s < self.n_components for s in state_sequence):
                raise ValueError(f"state_sequence bevat ongeldige waarden: {state_sequence}")
            if not all(0 <= x < self.n_features for x in X):
                raise ValueError(f"X bevat ongeldige waarden: {X}")
            
            log_prob = log(self.startprob_[state_sequence[0]])  # Startkans
            for t in range(1, len(state_sequence)):
                log_prob += log(self.transmat_[state_sequence[t-1], state_sequence[t]])  # Overgangskans
            
            for t in range(len(state_sequence)):
                emission_prob = self.emissionprob_[state_sequence[t], X[t]]
                if emission_prob <= 0:
                    raise ValueError(f"Ongeldige emissiekans: {emission_prob} bij t={t}, state={state_sequence[t]}, emission={X[t]}")
                log_prob += log(emission_prob)  # Emissiekans
            
            return log_prob

