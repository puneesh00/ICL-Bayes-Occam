import numpy as np
import os
os.environ['MPLCONFIGDIR'] = "/scratch/st-cthrampo-1/puneesh"

import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from scipy.special import logsumexp



def log_results(log_file, step, loss, results, string_list):
    # Iterate over each order in the log file dictionary
     for ord, file in log_file.items():
        # Start with the step and loss information
        log_line = f"Step {step + 1}, Loss: {loss:.4f}"
        
        # Loop over the string list and corresponding results[ord] dictionary
        result_entries = []
        #print(string_list)
        #print(results[ord].items())
        for key, value in results[ord].items():
            # Get the result from the results[ord] dictionary using the current key
            result_value = value
            name = string_list[key]
            # If result_value exists, append it to the result_entries list
            if result_value is not None:
                result_entries.append(f"KL {name}: {result_value:.4f}")
        
        # Join the result entries with commas and add to the log_line
        log_line += ", " + ", ".join(result_entries)
        
        # Write the log line to the file
        file.write(log_line + "\n")

        # Optionally flush after every write if you want to monitor logs
        # file.flush()

def open_log_files(order_list, save_path):
    # Dictionary to store opened file objects
    log_files_dict = {}
    
    # Iterate through each element in order_list and generate a log file
    for order in order_list:
        # Construct the file name dynamically with the order number
        log_filename = f'test_kl_divergence_log_order{order}.txt'
        
        # Join the path and file name
        full_path = os.path.join(save_path, log_filename)
        
        # Open the log file and store it in the dictionary
        log_files_dict[order] = open(full_path, "w")
        
    return log_files_dict

def log_attention_scores(log_files, step, avg_attn_scores):
    """
    Log the average attention scores for each block into separate files.
    
    Parameters:
    - log_files (dict): Dictionary where keys are block indices and values are file handles for each block.
    - step (int): The current step number.
    - avg_attn_scores (dict): Dictionary where keys are block names (e.g., "block_0") and values are the average attention scores.
    """
    # Iterate over each block and its corresponding average attention scores
    for block_name, attn_scores in avg_attn_scores.items():
        # Get the log file for the corresponding block
        log_file = log_files.get(block_name)
        
        if log_file is not None:
            # Create the log line with step info
            log_line = f"step {step + 1}, Attn scores: "
            
            # Convert the attention scores to a list of formatted strings (up to 4 decimals)
            attn_scores_str = [f"{score:.4f}" for score in attn_scores.flatten()]
            
            # Join all attention scores with commas and add to the log line
            log_line += ", ".join(attn_scores_str)
            
            # Write the log line to the file
            log_file.write(log_line + "\n")
            
            # Optionally flush after every write if you want to monitor logs
            # log_file.flush()

def save_model(model, save_path, step=None):
    """
    Save the trained model to a specified file path.
    
    Parameters:
    - model (nn.Module): The trained model to save.
    - save_path (str): The path where the model should be saved.
    - step (int, optional): If specified, include the step number in the filename.
    """
    #print("internal", save_path)
    if step is not None:
        save_path = os.path.join(save_path, f"step:{step}_model.pth")
    else:
        save_path = f"{save_path}.pth"
    
    # Save the model state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model_from_directory(directory_path, model, device):
    """
    Load a PyTorch model from a directory that contains a .pth file.
    
    Parameters:
    - directory_path (str): The path to the directory where the model file is located.
    - model_class (torch.nn.Module): The class of the model you want to load.
    
    Returns:
    - model (torch.nn.Module): The loaded model.
    """
    # Find the .pth file in the directory
    model_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.pth')])
    
    if not model_files:
        raise FileNotFoundError("No .pth file found in the specified directory.")
    
    # Assuming there is only one .pth file in the directory
    model_path = os.path.join(directory_path, model_files[-1])
    print(f"Loading model from {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location = device))
    
    return model




def combine_stats(stats, negll, nzr, test_order_list, V, lambda_, indices):
    combined_stats = {}
    
    for split in stats:
        combined_stats[split] = {}
        #log_T = 
        for order in test_order_list:
            # Extract the relevant tensors
            tensors = {idx: stats[split][order][idx] for idx in indices}

            #print("tensor selec", tensors[2][0:2], tensors[4][0:2])
            
            # Compute log likelihood: log_ll = -negll
            log_ll = {idx: -negll[split][order][idx] for idx in indices}

            # Compute costs
            costs = {idx: V**(idx-1) * (V - 1) for idx in indices}  # C_h = V^h * (V-1)
            
            # Compute log weights: log_weights = -lambda_ * costs
            log_weights = {idx: -np.log(T) * costs[idx] for idx in indices}

            # Compute log_prior using log-sum-exp for numerical stability
            log_weight_values = np.array(list(log_weights.values()))  # Shape: (num_indices,)
            #log_sum_weights = logsumexp(log_weight_values)  # Scalar
            log_prior = {idx: log_weights[idx] for idx in indices}

            #print("log_prior", log_prior)

            # Compute log_posterior: log_posterior = log_ll + log_prior
            log_posterior = {idx: log_ll[idx] + log_prior[idx] for idx in indices}  # Each entry shape: (N,)

            # Stack log_posterior values into a 2D array
            log_posterior_stack = np.stack(list(log_posterior.values()), axis=0)  # Shape: (num_indices, N)

            # Compute log_sum_posterior using log-sum-exp across indices (axis=0)
            log_sum_posterior = logsumexp(log_posterior_stack, axis=0)  # Shape: (N,)

            # Compute log_normalized_posterior: log_posterior - log_sum_posterior
            log_normalized_posterior = log_posterior_stack - log_sum_posterior  # Shape: (num_indices, N)

            # Convert log_normalized_posterior to normal scale
            normalized_posterior = np.exp(log_normalized_posterior)  # Shape: (num_indices, N)

            #print("posterior values", normalized_posterior)

            # Convert normalized_posterior to a PyTorch tensor and reshape for broadcasting
            normalized_posterior_tensor = torch.tensor(normalized_posterior).unsqueeze(-1)  # Shape: (num_indices, N, 1)

            # Stack tensors into a tensor of shape (num_indices, N, tensor_dim)
            tensor_stack = torch.stack([tensors[idx] for idx in indices], dim=0)  # Assuming tensors[idx] has shape (N, tensor_dim)

            # Compute the combined tensor as a weighted sum
            combined_tensor = torch.sum(normalized_posterior_tensor * tensor_stack, dim=0)  # Shape: (N, tensor_dim)

            #print("prediction", combined_tensor)
            combined_stats[split][order] = combined_tensor

    return combined_stats


def tensor_to_nested_dict_with_keys(M, keys_a, keys_b):
    """
    Convert a tensor M into a nested dictionary with custom keys.

    Parameters:
        M (numpy.ndarray): The input tensor of shape (A, B, C).
        keys_a (list): List of keys for the first dimension (length A).
        keys_b (list): List of keys for the second dimension (length B).

    Returns:
        dict: Nested dictionary with custom keys.
    """
    return {
        keys_a[a]: {keys_b[b]: M[a, b, :] for b in range(M.shape[1])}
        for a in range(M.shape[0])
    }

def tune_lam_prob(log_P_data_given_h, P_pred_given_h, P_true_pred, H, 
                 num_steps=1000, lr=5e-1, epsilon=1e-12, device='cpu'):
    """
    Tune prior probabilities P(h) to minimize KL divergence between true and model-predicted probabilities.

    Args:
        log_P_data_given_h (torch.Tensor): Log-likelihoods, shape (N, H)
        P_pred_given_h (torch.Tensor): Prediction probabilities, shape (N, H, K)
        P_true_pred (torch.Tensor): True prediction probabilities, shape (N, K)
        H (int): Number of hypotheses
        num_steps (int): Number of optimization steps
        lr (float): Learning rate for optimizer
        epsilon (float): Small value to prevent division by zero
        device (str): Device to perform computations on ('cpu' or 'cuda')

    Returns:
        torch.Tensor: Learned prior probabilities P(h), shape (H,)
    """
    # Convert to float64 and move to device
    log_P_data_given_h = log_P_data_given_h.to(dtype=torch.float64, device=device)/400
    P_pred_given_h = P_pred_given_h.to(dtype=torch.float64, device=device)
    P_true_pred = P_true_pred.to(dtype=torch.float64, device=device)

    # Initialize H parameters with small positive values using Softplus
    # To ensure positivity, we'll parameterize alpha_h via Softplus
    # Initialize log_alpha to small negative values to start with alpha_h ~ Softplus(0) = ~0.6931
    log_alpha = torch.full((H,), -1.0, dtype=torch.float64, device=device, requires_grad=True)

    # Define optimizer with a reduced learning rate
    optimizer = optim.Adam([log_alpha], lr=lr)

    # Validate input data
    assert not torch.isnan(P_pred_given_h).any(), "P_pred_given_h contains NaN"
    assert not torch.isinf(P_pred_given_h).any(), "P_pred_given_h contains Inf"
    assert not torch.isnan(P_true_pred).any(), "P_true_pred contains NaN"
    assert not torch.isinf(P_true_pred).any(), "P_true_pred contains Inf"

    # Enable anomaly detection for debugging (optional but recommended during development)
    torch.autograd.set_detect_anomaly(True)

    for step in range(1, num_steps +1):
        optimizer.zero_grad()

        # Compute alpha using Softplus to ensure positivity
        alpha = torch.nn.functional.softplus(log_alpha)  # Shape: (H,)

        # Compute P(h) = alpha / sum(alpha)
        P_h = alpha / (alpha.sum() + epsilon)  # Shape: (H,)

        # Compute log P(h)
        log_P_h = torch.log(alpha + epsilon) - torch.log(alpha.sum() + epsilon)  # Shape: (H,)

        # Compute log P(data_n) using log-sum-exp for numerical stability
        # log P(data_n) = logsumexp_h [ log P(data_n | h) + log P(h) ]
        log_P_data = torch.logsumexp(log_P_data_given_h + log_P_h.unsqueeze(0), dim=1)  # Shape: (N,)

        # Compute log posterior: log P(h | data_n) = log P(data_n | h) + log P(h) - log P(data_n)
        log_P_h_given_data = log_P_data_given_h + log_P_h.unsqueeze(0) - log_P_data.unsqueeze(1)  # Shape: (N, H)

        # Clamp to prevent underflow (optional but recommended)
        log_P_h_given_data = torch.clamp(log_P_h_given_data, min=-700)  # torch.exp(-700) ~ 5e-305

        # Compute posterior probabilities in probability space
        P_h_given_data = torch.exp(log_P_h_given_data)  # Shape: (N, H)

        #print("posterior nan", torch.isnan(P_pred_given_h).any())
        #print("posterior inf", torch.isinf(P_pred_given_h).any())
        # Normalize to ensure they sum to 1 for each sequence
        P_h_given_data = P_h_given_data / (P_h_given_data.sum(dim=1, keepdim=True) + epsilon)  # Shape: (N, H)
        #print("p h given data", torch.isnan(P_h_given_data).any())
        #print("p h given data inf ", torch.isinf(P_h_given_data).any())
        # Compute model's prediction: sum_h P(h | data_n) * P(pred_n | h)
        P_model_pred = torch.sum(P_h_given_data.unsqueeze(2) * P_pred_given_h, dim=1)  # Shape: (N, K)
        #print("p model pred", torch.isnan(P_model_pred).any())
        #print("p model pred inf", torch.isinf(P_model_pred).any())
        # Normalize P_model_pred to ensure it's a valid probability distribution
        P_model_pred = P_model_pred / (P_model_pred.sum(dim=1, keepdim=True) + epsilon)  # Shape: (N, K)
        #print("p model pred 2", torch.isnan(P_model_pred).any())
        #print("p model pred 2 inf", torch.isinf(P_model_pred).any())
        # Compute KL divergence for each sequence: KL(P_true || P_model)
        # KL(P || Q) = sum P * log(P / Q)

        ratio = (P_true_pred + epsilon) / (P_model_pred + epsilon)
        ratio = torch.clamp(ratio, min=epsilon, max=1e6)  # Cap at a reasonable upper bound
        kl_div = P_true_pred * torch.log(ratio)  # Shape: (N, K)

        #print("log ratio has nan", torch.isnan(torch.log((P_true_pred + epsilon) / (P_model_pred + epsilon))).any())
        #print("kl nan", torch.isnan(kl_div).any())
        #print("kl inf", torch.isinf(kl_div).any())

        #print((P_model_pred == 0).any())  # Check for zeros in P_model_pred

        kl_div = kl_div.sum(dim=1).mean()  # Scalar

        # Check for NaN or Inf in loss
        if torch.isnan(kl_div) or torch.isinf(kl_div):
            print(f"NaN or Inf detected at step {step}")
            break

        # Backpropagate
        kl_div.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_([log_alpha], max_norm=1.0)

        # Update parameters
        optimizer.step()

        # (Optional) Print loss every 100 steps and the first step
        if step ==1 or step %100 ==0:
            print(f"step {step}/{num_steps}, KL Divergence Loss: {kl_div.item():.6f}")

    # After training, retrieve the learned priors
    with torch.no_grad():
        alpha = torch.nn.functional.softplus(log_alpha)
        P_h_learned = alpha / (alpha.sum() + epsilon)
        print("\nLearned Priors P(h):")
        for h in range(H):
            print(f"P(h={h+1}) = {P_h_learned[h].item():.4e}")
    return P_h_learned


def visualize_and_save_attention(attention_scores, save_path, order):
    """
    Visualize and save attention scores for multiple heads.
    
    Parameters:
    - attention_scores (torch.Tensor or np.array): Tensor of shape (num_heads, T, T).
    - save_path (str): Directory path to save the heatmap.
    - order (int): The order to include in the filename.
    """
    

    if attention_scores.ndim==2:
        num_heads = 1
        T, _ = attention_scores.shape

    else:
        num_heads, T, _ = attention_scores.shape    

    # Create a figure with subplots for each head
    fig, axes = plt.subplots(1, num_heads, figsize=(8 * num_heads, 8))
    
    # Ensure `axes` is always iterable, even if there's only one subplot
    if num_heads == 1:
        axes = [axes]
    
    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attention_scores[-10:, -10:], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {i + 1}')
        ax.set_xlabel('Position in Sequence')
        ax.set_ylabel('Position in Sequence')
    
    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Attention Score')

    # Save the plot
    save_name = os.path.join(save_path, f"attention_heatmap_end_lay2_order{order}.png")
    plt.savefig(save_name, bbox_inches='tight')
    print(f"Heatmap saved to {save_name}")
    
    # Show the plot
    plt.show()
    plt.close(fig)
