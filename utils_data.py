import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from collections import Counter
from scipy.special import logsumexp



class MarkovChainGenerator:
    def __init__(self, vocab_size: int, order: int, max_transitions: int, 
                 seed: int):
        
        self.vocab_size = vocab_size
        self.order = order
        self.states = np.arange(self.vocab_size)
        self.seed = seed
        #np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        
        self.transition_matrix = self.generate_constrained_transition_matrix(self.vocab_size, self.order, 
                                                                                 max_transitions,
                                                                                 self.rng)

    @classmethod
    def generate_constrained_transition_matrix(cls, vocab_size: int, 
                                               order: int, max_transitions: int,
                                               rng: np.random.Generator) -> np.ndarray:
        """
        Generate a k-order Markov chain transition matrix where each state transitions to at most S<k states.
        
        :param vocab_size: Size of the vocabulary (V)
        :param order: Order of the Markov chain (k)
        :param max_transitions: Maximum number of possible transitions for each state (S)
        :return: Transition matrix of shape (V^k, V)
        """
        if max_transitions > vocab_size:
            raise ValueError("max_transitions must be less than vocab_size")

        num_states = vocab_size ** order
        transition_matrix = np.zeros((num_states, vocab_size))

        #transition_matrix = np.random.dirichlet(np.ones(vocab_size), size=vocab_size)
        for state in range(num_states):
            num_transitions = max_transitions
            
            # Randomly choose which states to transition to
            transition_indices = rng.choice(vocab_size, num_transitions, replace=False)
            
            # Assign random probabilities to these transitions
            probabilities = rng.dirichlet(np.ones(num_transitions))
            
            # Fill in the transition matrix
            transition_matrix[state, transition_indices] = probabilities 

        return transition_matrix

    def generate_sequence(self, length: int, start=None) -> List[int]:
        # Start with a random state sequence of length 'order'

        if start is not None:
            start = np.real(start)
            state = np.random.choice(np.arange(self.vocab_size**self.order), p=start)
            sequence = self._index_to_state(state)
        else:
             sequence = list(np.random.choice(self.states, size=self.order))
        
        for _ in range(length - self.order):
            current_state = tuple(sequence[-self.order:])
            state_index = self._state_to_index(current_state)
            #print('Current', current_state)
            #print("state", state_index)
            #print("row for transitions", self.transition_matrix[state_index])
            next_state = np.random.choice(self.states, p=self.transition_matrix[state_index])
            sequence.append(next_state)
        
        return sequence

    def _state_to_index(self, state: Tuple[int, ...]) -> int:
        """Convert a state tuple to its corresponding row index in the transition matrix."""
        return sum(s * (self.vocab_size ** i) for i, s in enumerate(reversed(state)))

    def _index_to_state(self, index):
        state = []
        for _ in range(self.order):
            state.append(index % self.vocab_size)
            index //= self.vocab_size
        return list(reversed(state))
    
    def generate_multiple_sequences(self, num_sequences: int, length: int, window_size: int 
                                    ) -> np.ndarray:
        data = []
        for _ in range(num_sequences):
            sequence = self.generate_sequence(length)
            data.append(sequence)
    
        return np.asarray(data)
    @staticmethod
    def test_data_train(self, generators, num_sequences: int, length: int):
        data = []
        for i in range(len(generators)):
            data_gr = generators[i].generate_multiple_sequences(num_sequences, length)
            data.append(data_gr)

        data = np.asarray(np.squeeze(data_gr))

        return data    

    @staticmethod
    def generate_data_target_pairs(sequence: List[int], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        target = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            data.append(window[:-1])
            target.append(window[1:])
        return np.array(data), np.array(target)
    

    def stationary_distribution_order_k_markov(self):
        """
        Calculate the stationary distribution of an order-k Markov chain.
        
        Args:
        transition_matrix (numpy.ndarray): Transition matrix of shape (V^k, V)
        k (int): Order of the Markov chain
        V (int): Size of the vocabulary (number of possible states for each position)
        
        Returns:
        numpy.ndarray: Stationary distribution of shape (V^k,)
        """
        # Ensure the transition matrix is the correct shape
        V = self.vocab_size
        k = self.order
        assert self.transition_matrix.shape == (V**k, V), "Transition matrix shape mismatch"
        
        # Create the full transition matrix
        self.full_transition = np.zeros((V**k, V**k))
        
        for i in range(V**k):
            for j in range(V):
                next_state = (i * V + j) % (V**k)
                self.full_transition[next_state, i] = self.transition_matrix[i, j]
        
        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self.full_transition.T)
        
        # Find the index of the eigenvalue closest to 1
        index = np.argmin(np.abs(eigenvalues - 1))
        
        # Get the corresponding eigenvector and normalize it
        stationary = eigenvectors[:, index].real
        stationary /= np.sum(stationary)
        
        return stationary

class SequenceAnalyzer:
    @staticmethod
    def get_n_gram_statistics(sequences: torch.Tensor, n: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute n-gram statistics for each sequence separately.
        
        :param sequences: List of integer sequences
        :param n: The order of n-gram
        :param vocab_size: Size of the vocabulary
        :return: n_gram_stats (probability distributions), n_gram_probs, pi_counts
        """
        n_gram_stats = []
        num_seq, seq_length = sequences.shape

        #device = sequences.device
        n_gram_stats = torch.zeros((num_seq, vocab_size))
        n_gram_probs = torch.zeros((num_seq, vocab_size ** n))
        pi_counts = torch.zeros((num_seq, int(vocab_size ** (n - 1))))

        if n==0:
            n_gram_stats = torch.ones(vocab_size)/vocab_size

        elif n==1:
            for i in range(vocab_size):
                # Count occurrences of each value i in each row
                #n_gram_counts = torch.zeros((num_seq, vocab_size), device=device)
                n_gram_stats[:, i] = (sequences == i).sum(dim=1)
            n_gram_stats = n_gram_stats / (torch.sum(n_gram_stats, axis=1, keepdims=True) + 1e-12)

        else:
            for s in range(num_seq):
                # Initialize n-gram counts array
                sequence = sequences[s]
                n_gram_counts = torch.zeros([vocab_size] * n)
                pi_counts_seq = torch.zeros([vocab_size] * (n - 1))

                # Count n-grams in the sequence
                for i in range(seq_length - n + 1):
                    n_gram = tuple(sequence[i:i+n].tolist())
                    prefix = tuple(sequence[i:i + n - 1].tolist())  # Extract prefix of length (n-1)

                    n_gram_counts[n_gram] += 1
                    pi_counts_seq[prefix] += 1 

                # Convert counts to probabilities

                # Convert counts to probabilities
                total_n_grams = torch.sum(n_gram_counts, axis=-1, keepdims=True) + 1e-12
                n_gram_probs_seq = n_gram_counts / total_n_grams
                #n_gram_probs = n_gram_counts / (torch.sum(n_gram_counts, axis=-1, keepdims=True) + 1e-12)
                last_n = tuple(sequence[-(n-1):].tolist())
                n_gram_stats[s] = n_gram_probs_seq[last_n]
                n_gram_probs[s] = n_gram_probs_seq.view(-1)  # Flatten to 1D

                pi_counts[s] = pi_counts_seq.view(-1) / (torch.sum(pi_counts_seq) + 1e-12)  # Flatten and normalize
            
        return n_gram_stats, n_gram_probs, pi_counts
    
    @staticmethod
    def compute_entropy(n_gram_probs: torch.Tensor, pi_counts: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """
        Compute the entropy using n-gram probabilities and (n-1)-gram context counts.
        
        :param n_gram_probs: Tensor of shape (num_seq, vocab_size^n) containing n-gram probabilities for each sequence.
        :param pi_counts: Tensor of shape (num_seq, vocab_size^(n-1)) containing (n-1)-gram context probabilities.
        :param vocab_size: Size of the vocabulary.
        :return: Tensor containing the entropy for each sequence.
        """
        num_seq = n_gram_probs.shape[0]
        
        # Reshape n_gram_probs to (num_seq, vocab_size^(n-1), vocab_size)
        n_gram_probs_reshaped = n_gram_probs.view(num_seq, -1, vocab_size)
        
        # Ensure pi_counts has shape (num_seq, vocab_size^(n-1)) for consistency
        assert pi_counts.shape == (num_seq, n_gram_probs_reshaped.shape[1])
        
        # Compute the entropy term: -pi * p * log(p)
        entropy = -torch.sum(pi_counts.unsqueeze(-1) * n_gram_probs_reshaped * 
                            torch.log(n_gram_probs_reshaped + 1e-12), dim=(1, 2))
        
        return entropy
    
    @staticmethod
    def compute_avg_neg_log_likelihood(sequences, q_tensors, order, vocab_size):
        """
        Compute the average negative log-likelihood for a batch of sequences, each with its own transition matrix.
        
        :param sequences: 2D array of token indices (shape: [num_sequences, sequence_length])
        :param q_tensors:  List of transition matrices, one for each sequence (shape: [num_sequences, vocab_size^order, vocab_size])
        :param order: Order of the Markov model (e.g., 1 for order-1, 3 for order-3)
        :return: Array of average negative log-likelihoods, one per sequence
        """
        num_sequences, T = sequences.shape

        #print("num_seq", num_sequences)
        #print("vocab size", vocab_size)
        avg_neg_log_likelihoods = []

        #print(q_tensors.shape)
        q_tensors = q_tensors.view(num_sequences, -1, vocab_size)
        #print("reshape",q_tensors.shape)
        for s in range(num_sequences):
            seq = sequences[s]
            q = q_tensors[s]  # Transition matrix specific to this sequence
            neg_log_likelihoods = []
            
            # Iterate over the sequence, starting from `order` to use `order` previous tokens as context
            for t in range(order, T):
                # Get the previous `order` tokens as context

                if order>0:
                    prev_tokens = tuple(seq[t-order:t])
                    state = sum(s * (vocab_size ** i) for i, s in enumerate(reversed(prev_tokens)))
                current_token = seq[t]
                
                
                #print((state, current_token))
                # Retrieve the transition probability for the current token given the previous context
                if order>0:
                    prob = q[(state, current_token)]
                else:
                    print(q.shape)
                    prob = q.squeeze()[current_token]    
                
                # Compute the negative log-likelihood
                neg_log_likelihoods.append(-torch.log(prob + 1e-12))  # Add a small value to avoid log(0)
            
            # Compute the average negative log-likelihood for this sequence
            avg_neg_log_likelihood = torch.sum(torch.Tensor(neg_log_likelihoods)) #sum, not avg.
            avg_neg_log_likelihoods.append(avg_neg_log_likelihood)
        
        return torch.Tensor(avg_neg_log_likelihoods)


    @staticmethod
    def get_unigram_statistics(sequences: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """
        Compute unigram statistics for a list of sequences.
        
        :param sequences: List of integer sequences
        :param vocab_size: Size of the vocabulary
        :return: Unigram probability distribution
        """
        return SequenceAnalyzer.get_n_gram_statistics(sequences, 1, vocab_size)

    @staticmethod
    def calculate_kl_divergence(predictions: torch.Tensor, target_distribution: torch.Tensor, pred) -> float:
        """
        Calculate the KL divergence between predictions and target distribution.
        
        :param predictions: Predicted probability distribution
        :param target_distribution: Target probability distribution
        :return: KL divergence
        """
        # Ensure the distributions sum to 1
        epsilon = 1e-12

        if not(pred):
            predictions = F.softmax(predictions, dim=-1).detach().cpu().numpy() + epsilon
        else:
            predictions = predictions.detach().cpu().numpy() + epsilon    
        #predictions = predictions.cpu().numpy() + epsilon
        #print("probs", predictions)
        # Add a small epsilon to avoid log(0)
        
        target_distribution = target_distribution.cpu().numpy() + epsilon
        #predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        #target_distribution = np.clip(target_distribution.cpu().numpy(), epsilon, 1.0 - epsilon)
        
        # Calculate KL divergence
        kl_div = np.mean(np.sum(target_distribution * np.log(target_distribution / predictions), axis=-1))
        
        return kl_div

    @staticmethod
    def get_perplexity(sequence: List[int], n_gram_model: np.ndarray) -> float:
        """
        Calculate the perplexity of a sequence given an n-gram model.
        
        :param sequence: Input sequence
        :param n_gram_model: n-gram probability distribution
        :return: Perplexity
        """
        n = n_gram_model.ndim
        log_likelihood = 0
        for i in range(len(sequence) - n + 1):
            n_gram = tuple(sequence[i:i+n])
            probability = n_gram_model[n_gram]
            log_likelihood += np.log2(probability)
        
        perplexity = 2 ** (-log_likelihood / (len(sequence) - n + 1))
        return perplexity


class MarkovChainEvaluator:
    def __init__(self, vocab_size: int, max_transitions: int, orders: List[int], num_sequences: int, 
                 sequence_length: List[int], t_grams: List[int], 
                 seed: int):
        self.vocab_size = vocab_size
        self.max_transitions = max_transitions
        self.orders = orders
        self.max_transitions = max_transitions
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.t_grams = t_grams
        self.sequence_analyzer = SequenceAnalyzer()
        self.generators = {}
        self.seed = seed

        for order in self.orders:
            self.generators[order] = MarkovChainGenerator(vocab_size=self.vocab_size, order = order, 
                                            max_transitions=self.max_transitions, seed=self.seed) 
                    
    def generate_sequences_for_orders(self, base_seed, stationary : bool = False) -> Dict[int, torch.Tensor]:
        sequences_by_order = {}
        start = {}
        for idx, order in enumerate(self.orders):
            generator = self.generators[order]
            stationary_dist = generator.stationary_distribution_order_k_markov()
            sequences = []
            for p in range(self.num_sequences):
                #seed = base_seed + p
                #np.random.seed(seed)
                start = stationary_dist if stationary else None
                sequences.append(generator.generate_sequence(self.sequence_length[idx],start))
            # sequences = [generator.generate_sequence(self.sequence_length) 
            #              for _ in range(self.num_sequences)]
            sequences_tensor = torch.tensor(sequences, dtype=torch.long)
            sequences_by_order[order] = sequences_tensor
        return sequences_by_order

    def compute_t_gram_stats(self, sequences: torch.Tensor, ord: int) -> Dict[int, torch.Tensor]:
        t_gram_stats = {}
        t_gram_probs = {}
        p_counts_seq = {}

        for t in self.t_grams:
            if t>0:
                n_gram_stats, n_gram_probs, pi_counts = self.sequence_analyzer.get_n_gram_statistics(sequences, t-1, self.vocab_size)
                t_gram_stats[t] = n_gram_stats
                t_gram_probs[t] = n_gram_probs
                p_counts_seq[t] = pi_counts

        stats = []
        for i in range(sequences.shape[0]):
            index = self.generators[ord]._state_to_index(sequences[i,-ord:]) 
            probs = self.generators[ord].transition_matrix[index]
            stats.append(probs)
        
        t_gram_stats[0] = torch.tensor(stats) 
        t_gram_probs[0] = n_gram_probs
        p_counts_seq[0] = pi_counts    
            
        return t_gram_stats, t_gram_probs, p_counts_seq

    # def evaluate_kl_divergence(self, predictions: np.ndarray, t_gram_stats: Dict[int, np.ndarray]) -> Dict[int, float]:
    #     kl_divergences = {}
    #     for t, stats in t_gram_stats.items():
    #         kl_div = self.sequence_analyzer.calculate_kl_divergence(predictions, stats)
    #         kl_divergences[t] = kl_div
    #     return kl_divergences
    @staticmethod
    def run_evaluation(predictions: List[torch.Tensor], t_gram_stats: List[Dict[int, torch.Tensor]],
                       t_grams, pred=False, evals=10) -> Dict[int, float]:
        
        results = {}
        for t in t_grams:
            if evals>1:
                kl_divergences = 0
                for pred_tensor, tgs_dict in zip(predictions, t_gram_stats):
                    kl_div = SequenceAnalyzer.calculate_kl_divergence(pred_tensor, tgs_dict[t], pred)
                    kl_divergences += kl_div
                results[t] = kl_divergences / evals
            else:
                results[t] = SequenceAnalyzer.calculate_kl_divergence(predictions, t_gram_stats[t], pred) 

        return results 



def generate_evaluation_seqs(order_list, num_tm, vocab_size, max_transitions, t_gram_list, test_seq_len):
        t_gram_stats = {}

        test_seqs_list = {}
        t_gram_stats, test_seqs_list = {ordr: [] for ordr in order_list}, {ordr: [] for ordr in order_list}

        for j in range(num_tm):
            seed = 12574*j + 5
            evaluator = MarkovChainEvaluator(vocab_size=vocab_size, max_transitions=max_transitions, orders=order_list, num_sequences=100, 
                                             sequence_length=test_seq_len, t_grams=t_gram_list, 
                                             seed=seed)
            
            test_sequences = evaluator.generate_sequences_for_orders(base_seed=None, stationary=True) 

            for ordr in order_list:
                test_seqs_list[ordr].append(test_sequences[ordr])
                tgs,_,_ = evaluator.compute_t_gram_stats(test_sequences[ordr], ordr) # Dict[int, Dict[int, torch.Tensor]]
                t_gram_stats[ordr].append(tgs)

        return {"seqs": test_seqs_list, 
                "tgram_stats": t_gram_stats}

def run_eval(test_order_list, test_seqs_list, device, transformer, 
             t_gram_stats, t_grams, step):

    results = {}

    with torch.no_grad():
        
        for ordr in test_order_list:
            test_outputs_list = []

            evals = len(test_seqs_list[ordr])
            eval_list = range(evals)

            for k in eval_list:
                test_data = test_seqs_list[ordr][k].to(device)
                test_outputs,_ = transformer(test_data)
            
                test_outputs_list.append(test_outputs[:,-1,:])
                
            results[ordr] = MarkovChainEvaluator.run_evaluation(test_outputs_list, t_gram_stats[ordr], t_grams=t_grams,
                                                     evals=len(eval_list))   #Dict[int (ord), Dict[int (t-gram), float]]
            kl_values_str = ', '.join(f"{key}: {value:.8f}" for key, value in results[ordr].items())
            
            print(f"Itr [{step}], Order: {ordr}, KL values: {kl_values_str}\n")
            
    return results