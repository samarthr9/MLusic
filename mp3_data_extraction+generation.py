import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import soundfile as sf

file_name_template = '/Users/samarthrao/Downloads/Todi_Audio_Files/KVN_todi_1/phrases/phrase_%s.mp3'
features = []

# Function to pad audio
def pad_audio(audio, target_length):
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), 'constant')
    return audio

# Load phrase_11 to determine the target length
file_name_longest = '/Users/samarthrao/Downloads/Todi_Audio_Files/KVN_todi_1/phrases/phrase_11.mp3'
y_longest, sr_longest = librosa.load(file_name_longest, sr=None)
target_length = len(y_longest)
print(f"Target length in samples (from phrase_11): {target_length}")

# Load, pad, and extract features
for i in range(1, 21):
    file_name = file_name_template % str(i)
    y_current, sr_current = librosa.load(file_name, sr=None)
    y_current = pad_audio(y_current, target_length)

    f0_current, voiced_flag_current, voiced_probs_current = librosa.pyin(y_current, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    chroma_current = librosa.feature.chroma_cens(y=y_current, sr=sr_current)
    spectral_centroids_current = librosa.feature.spectral_centroid(y=y_current, sr=sr_current)
    spectral_contrast_current = librosa.feature.spectral_contrast(y=y_current, sr=sr_current)
    onset_env_current = librosa.onset.onset_strength(y=y_current, sr=sr_current)
    tempo_current, beats_current = librosa.beat.beat_track(onset_envelope=onset_env_current, sr=sr_current)

    features.append({
        'f0': f0_current,
        'chroma': chroma_current,
        'spectral_centroids': spectral_centroids_current,
        'spectral_contrast': spectral_contrast_current,
        'tempo': tempo_current,
        'beats': beats_current
    })

# Determine maximum lengths for each feature type
max_len_f0 = max(len(np.nan_to_num(f['f0']).flatten()) for f in features)
max_len_chroma = max(f['chroma'].flatten().shape[0] for f in features)
max_len_spectral_centroids = max(f['spectral_centroids'].flatten().shape[0] for f in features)
max_len_spectral_contrast = max(f['spectral_contrast'].flatten().shape[0] for f in features)
max_len_beats = max(len(f['beats']) for f in features)

# Function to pad feature arrays
def pad_features(feature_array, max_length):
    if len(feature_array) < max_length:
        return np.pad(feature_array, (0, max_length - len(feature_array)), 'constant')
    return feature_array[:max_length]

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = self.features[idx]
        f0 = pad_features(np.nan_to_num(item['f0']).flatten(), max_len_f0)
        chroma = pad_features(item['chroma'].flatten(), max_len_chroma)
        spectral_centroids = pad_features(item['spectral_centroids'].flatten(), max_len_spectral_centroids)
        spectral_contrast = pad_features(item['spectral_contrast'].flatten(), max_len_spectral_contrast)
        tempo = item['tempo']
        beats = pad_features(item['beats'], max_len_beats)
        
        # Stack or concatenate the features as necessary
        data = np.hstack([
            f0,
            chroma,
            spectral_centroids,
            spectral_contrast,
            [tempo],
            beats
        ])

        return torch.tensor(data, dtype=torch.float32)

# LSTM Model for sequence prediction
class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AudioLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Prepare the data loader and train the model:
# Hyperparameters
input_size = max_len_f0 + max_len_chroma + max_len_spectral_centroids + max_len_spectral_contrast + 1 + max_len_beats
hidden_size = 128
num_layers = 2
output_size = input_size  # Predicting the next chunk of similar feature size
num_epochs = 20
batch_size = 4
learning_rate = 0.001

# Create dataset and dataloader
dataset = AudioDataset(features)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = AudioLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for data in dataloader:
        data = data.unsqueeze(1)  # Adding sequence dimension for LSTM (batch_size, sequence_length, input_size)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, data.squeeze(1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction code chunk
# Ensure the model is in evaluation mode
model.eval()

# Use the features from the 15th Todi audio file as input for prediction
input_features = features[14]  # 15th file (index 14)
input_data = np.hstack([
    pad_features(np.nan_to_num(input_features['f0']).flatten(), max_len_f0),
    pad_features(input_features['chroma'].flatten(), max_len_chroma),
    pad_features(input_features['spectral_centroids'].flatten(), max_len_spectral_centroids),
    pad_features(input_features['spectral_contrast'].flatten(), max_len_spectral_contrast),
    [input_features['tempo']],
    pad_features(input_features['beats'], max_len_beats)
])
input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Adding batch and sequence dimensions

# Predict the next chunk
with torch.no_grad():
    predicted_chunk = model(input_data)

# Print the predicted chunk
print(predicted_chunk)

# Save the predicted chunk tensor to a file
torch.save(predicted_chunk, '/Users/samarthrao/Downloads/Todi_Audio_Files/KVN_todi_1/predicted_chunk.pt')
