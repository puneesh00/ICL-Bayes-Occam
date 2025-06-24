import os
os.environ['MPLCONFIGDIR'] = "/scratch/st-cthrampo-1/puneesh"

import matplotlib.pyplot as plt
import numpy as np

import argparse
import re  # Import the regex library to handle parsing
import json
import ast


parser = argparse.ArgumentParser(description="Training script for transformer model.")

parser.add_argument('--path', type=str, required=True, help="Path to save file")
parser.add_argument('--order', type=int, required=True, help="Order")

args = parser.parse_args()

save_path = args.path
order = args.order
# Load the data from the log file
def load_data(filename):
    epochs = []
    loss = []
    kl_uniform = []
    kl_unigram = []
    kl_bigram = []
    kl_trigram = []
    kl_tetragram = []
    
    with open(filename, "r") as f:
        for line in f:
            if "Epoch" in line:
                # Extract the relevant data
                parts = line.split(',')
                epoch = int(parts[0].split()[1])
                l = float(parts[1].split()[1])
                kl_u = float(parts[2].split()[2])
                kl_ug = float(parts[3].split()[2])
                kl_bg = float(parts[4].split()[2])
                kl_tg = float(parts[5].split()[2])
                #kl_tetrag = float(parts[6].split()[2])

                #print(kl_u,kl_ug,kl_bg)
                epochs.append(epoch)
                loss.append(l)
                kl_uniform.append(kl_u)
                kl_unigram.append(kl_ug)
                kl_bigram.append(kl_bg)
                kl_trigram.append(kl_tg)
                #kl_tetragram.append(kl_tetrag)
    
    return np.array(epochs), np.array(loss), np.array(kl_uniform), np.array(kl_unigram), np.array(kl_bigram), np.array(kl_trigram), np.array(kl_tetragram)

def load_data1(filename, skip_lines):
    epochs = []
    losses = []
    uniform = []
    unigram = []
    bigram = []
    trigram = []

    total_epochs = 0
    
    with open(filename, "r") as f:
        # Skip the specified number of lines
        for _ in range(skip_lines):
            next(f)
        
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            epoch_line = lines[i].strip()
            kl_line = lines[i+1].strip() if i+1 < len(lines) else None
            
            if "Epoch" in epoch_line and kl_line and kl_line.startswith('{'):
                epoch_match = re.search(r'Epoch \[(\d+)/(\d+)\], Loss: ([\d.]+)', epoch_line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))
                    loss = float(epoch_match.group(3))
                    
                    try:
                        kl_data = ast.literal_eval(kl_line)
                        
                        epochs.append(epoch)
                        losses.append(loss)
                        uniform.append(float(kl_data[2][0]))
                        unigram.append(float(kl_data[2][1]))
                        bigram.append(float(kl_data[2][2]))
                        trigram.append(float(kl_data[2][3]))
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing KL data: {e}")
                        print(f"Problematic line: {kl_line}")
                else:
                    print(f"Skipping malformed epoch line: {epoch_line}")
            else:
                print(f"Skipping malformed data pair:\n{epoch_line}\n{kl_line}")
    
    return np.array(epochs), np.array(losses), np.array(uniform), np.array(unigram), np.array(bigram), np.array(trigram), total_epochs

# Plotting the KL Divergence over time
def plot_kl_divergence(filename, save_path=os.path.join(save_path, 'kl_divergence_plot_order2.png')):

    #epochs, _, kl_uniform, kl_unigram, kl_bigram = load_data(filename) 
    epochs, _, kl_uniform, kl_unigram, kl_bigram, kl_trigram, kl_tetragram = load_data(filename)
    print("THIS IS IT",kl_uniform, kl_unigram)
    #print("uni",kl_uniform)
    #print("ug",kl_unigram)
    #print("epoch",epochs)
    # Plot the KL divergences
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, kl_uniform, label='Uniform', color='blue')
    plt.plot(epochs, kl_unigram, label='Unigram', color='orange')
    plt.plot(epochs, kl_bigram, label='Bigram', color='green')
    plt.plot(epochs, kl_trigram, label='Trigram', color='violet')
    #plt.plot(epochs, kl_tetragram, label='Tetragram', color='red')
    #plt.plot(epochs, loss, label='Loss', color='black')
    # Shade the regions based on the minimum KL divergence
    #min_kl = np.minimum(np.minimum(kl_uniform, kl_unigram), kl_bigram)
    min_kl = np.minimum(np.minimum(np.minimum(kl_uniform, kl_unigram), kl_bigram),kl_trigram)
    #min_kl = np.minimum(min_kl,kl_tetragram)
    min_idx = np.argmin([kl_uniform, kl_unigram, kl_bigram, kl_trigram], axis = 0)
    #min_idx = np.argmin([kl_uniform, kl_unigram, kl_bigram, kl_trigram, kl_tetragram], axis=0)
    #min_idx = np.argmin([kl_uniform, kl_unigram, kl_bigram, kl_trigram, kl_tetragram], axis=0)
    
    plt.fill_between(epochs, 0, min_kl, where=(min_idx == 0), color='blue', alpha=0.1, interpolate=True)
    plt.fill_between(epochs, 0, min_kl, where=(min_idx == 1), color='orange', alpha=0.1, interpolate=True)
    plt.fill_between(epochs, 0, min_kl, where=(min_idx == 2), color='green', alpha=0.1, interpolate=True)
    plt.fill_between(epochs, 0, min_kl, where=(min_idx == 3), color='violet', alpha=0.1, interpolate=True)
    #plt.fill_between(epochs, 0, min_kl, where=(min_idx == 4), color='red', alpha=0.1, interpolate=True)
    
    # Set plot labels and title
    plt.ylim((0, 0.9))
    plt.xlabel("Epochs")
    plt.ylabel("KL-Div(Distribution||Model)")
    plt.title("Transformer KL-Divergence: 3 Symbols")
    plt.legend(loc='upper right')
    
    # Save the plot as an image file
    plt.savefig(save_path, format='png', dpi=300)  # Save as PNG with 300 DPI
    
    # Display the plot
    plt.show()

# Usage
full_path = os.path.join(save_path, 'kl_divergence_log_order2.txt')
plot_kl_divergence(full_path)