import numpy as np

class HiddenMarkovModel:
    def __init__(self, n_components, n_features):
        """
        Initialiseert het Hidden Markov Model.

        Parameters:
        n_components: aantal toestanden
        n_features: aantal mogelijke emissies
        """
        self.n_components = n_components
        self.n_features = n_features
        self.startprob_ = np.ones(n_components) / n_components  # Uniforme verdeling standaard
        self.transmat_ = np.ones((n_components, n_components)) / n_components  # Uniforme overgangswaarschijnlijkheden
        self.emissionprob_ = np.ones((n_components, n_features)) / n_features  # Uniforme emissiekansen

    def __str__(self):
        """
        Stringrepresentatie van het model.
        """
        return (f"HiddenMarkovModel(n_components={self.n_components}, n_features={self.n_features})\n"
                f"Start probabilities:\n{self.startprob_}\n"
                f"Transition matrix:\n{self.transmat_}\n"
                f"Emission probabilities:\n{self.emissionprob_}")

    def __repr__(self):
        """
        Officiële representatie van het model.
        """
        return self.__str__()

    def sample(self, n_samples):
        """
        Genereert toestanden en waarnemingen.

        Parameters:
        n_samples: aantal samples dat gegenereerd moet worden.

        Returns:
        (emissions, states): tuple van emissies en toestanden
        """
        states = []
        emissions = []

        # Start met een beginstaat op basis van de startprob_
        current_state = np.random.choice(self.n_components, p=self.startprob_)
        for _ in range(n_samples):
            states.append(current_state)

            # Kies een emissie gebaseerd op de huidige toestand
            emission = np.random.choice(self.n_features, p=self.emissionprob_[current_state])
            emissions.append(emission)

            # Kies de volgende toestand
            current_state = np.random.choice(self.n_components, p=self.transmat_[current_state])

        return np.array(emissions), np.array(states)
