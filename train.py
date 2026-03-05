"""
train.py – Zebra Finch Call-Type Classification
Trains CNN and RNN on log-mel spectrograms and saves checkpoints.

Usage:
    python train.py

Outputs (all under ./models/):
    cnn_checkpoint_ep{N}.pt   – checkpoint every 5 epochs
    rnn_checkpoint_ep{N}.pt   – checkpoint every 5 epochs
    cnn_best.pt               – best model (lowest val loss)
    rnn_best.pt               – best model (lowest val loss)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'PyTorch {torch.__version__}  |  torchaudio {torchaudio.__version__}')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path('recordings')
MODELS_DIR  = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

# ── Audio / spectrogram constants ─────────────────────────────────────────────
TARGET_SR   = 22050
CLIP_LENGTH = 1.0
N_SAMPLES   = int(TARGET_SR * CLIP_LENGTH)

N_FFT       = 1024
HOP_LENGTH  = 256
N_MELS      = 64
F_MIN       = 500
F_MAX       = 10_500

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE        = 32
EPOCHS            = 60
LR                = 1e-3
PATIENCE          = 10
CHECKPOINT_EVERY  = 5   # save a checkpoint every N epochs

# ── Class map ─────────────────────────────────────────────────────────────────
CLASS_MAP = {
    'Ag': 'Aggressive',
    'Be': 'Begging',
    'DC': 'Distance',
    'Di': 'Distress',
    'LT': 'LongTonal',
    'Ne': 'Nest',
    'So': 'Song',
    'Te': 'Tet',
    'Th': 'Thuk',
    'Tu': 'Tuck',
    'Wh': 'Whine',
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def extract_vt_prefix(filename: str) -> str | None:
    stem = Path(filename).stem
    m = re.match(r'^[^_]+_\d{6}[-_](.+)-[^-]+$', stem)
    if m:
        return m.group(1)[:2]
    return None


def load_dataframe() -> tuple[pd.DataFrame, LabelEncoder]:
    records = []
    for fname in sorted(DATA_DIR.iterdir()):
        if fname.suffix.lower() != '.wav':
            continue
        prefix = extract_vt_prefix(fname.name)
        if prefix is None or prefix not in CLASS_MAP:
            continue
        records.append({'path': str(fname), 'prefix': prefix,
                        'label': CLASS_MAP[prefix]})

    df = pd.DataFrame(records)
    le = LabelEncoder()
    df['class_id'] = le.fit_transform(df['label'])
    print(f'Total usable files: {len(df)}')
    print('Classes:', list(le.classes_))
    return df, le


def make_splits(df: pd.DataFrame):
    train_df, tmp_df = train_test_split(
        df, test_size=0.30, stratify=df['class_id'], random_state=SEED
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, stratify=tmp_df['class_id'], random_state=SEED
    )
    print(f'Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ZebraFinchDataset(Dataset):
    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
    )
    amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

    def __init__(self, dataframe: pd.DataFrame, augment: bool = False):
        self.df      = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_and_fix(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if sr != TARGET_SR:
            waveform = T.Resample(sr, TARGET_SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        n = waveform.shape[-1]
        if n < N_SAMPLES:
            waveform = F.pad(waveform, (0, N_SAMPLES - n))
        else:
            waveform = waveform[:, :N_SAMPLES]
        return waveform

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        mel     = self.mel_transform(waveform)
        log_mel = self.amplitude_to_db(mel)
        lo, hi  = log_mel.min(), log_mel.max()
        if hi > lo:
            log_mel = (log_mel - lo) / (hi - lo)
        return log_mel

    def _augment(self, spec: torch.Tensor) -> torch.Tensor:
        spec = T.FrequencyMasking(freq_mask_param=8)(spec)
        spec = T.TimeMasking(time_mask_param=12)(spec)
        return spec

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        wave  = self._load_and_fix(row['path'])
        spec  = self._spectrogram(wave)
        if self.augment:
            spec = self._augment(spec)
        label = torch.tensor(row['class_id'], dtype=torch.long)
        return spec, label


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class ZebraFinchCNN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1,  32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.encoder(x)))

    def features(self, x):
        return self.pool(self.encoder(x)).flatten(1)


class ZebraFinchRNN(nn.Module):
    def __init__(self, n_classes: int,
                 hidden: int = 128, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_MELS,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)   # (B, T, N_MELS)
        out, _ = self.lstm(x)
        feat   = out.mean(dim=1)
        return self.classifier(feat)

    def features(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out.mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = np.inf
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, name):
    path = MODELS_DIR / f'{name}_checkpoint_ep{epoch:03d}.pt'
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss':             val_loss,
    }, path)
    print(f'  Checkpoint saved → {path}')


def train_model(model, name, train_loader, val_loader,
                class_weights_tensor, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper   = EarlyStopping(patience=PATIENCE)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = eval_epoch(model,  val_loader,   criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f'[{name}] Ep {epoch:3d}  '
                  f'train loss={tr_loss:.4f} acc={tr_acc:.3f}  '
                  f'val loss={vl_loss:.4f} acc={vl_acc:.3f}')

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, vl_loss, name)

        if stopper.step(vl_loss, model):
            print(f'[{name}] Early stop at epoch {epoch}.')
            break

    stopper.load_best(model)
    print(f'[{name}] Best val loss: {stopper.best_loss:.4f}')

    # ── Save final best weights ───────────────────────────────────────────────
    best_path = MODELS_DIR / f'{name}_best.pt'
    torch.save(model.state_dict(), best_path)
    print(f'[{name}] Best model saved → {best_path}')

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Data
    df, le = load_dataframe()
    train_df, val_df, test_df = make_splits(df)

    N_CLASSES = len(le.classes_)

    train_ds = ZebraFinchDataset(train_df, augment=True)
    val_ds   = ZebraFinchDataset(val_df,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # Class weights
    train_counts       = train_df['class_id'].value_counts().sort_index()
    class_weights      = 1.0 / train_counts.values.astype(np.float32)
    class_weights      = class_weights / class_weights.sum() * N_CLASSES
    class_weights_t    = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Save label encoder classes so the notebook can restore it
    np.save(MODELS_DIR / 'label_classes.npy', le.classes_)
    print(f'Label classes saved → {MODELS_DIR / "label_classes.npy"}')

    # Train CNN
    print('\n=== Training CNN ===')
    cnn = ZebraFinchCNN(N_CLASSES).to(device)
    cnn_history = train_model(cnn, 'cnn', train_loader, val_loader,
                              class_weights_t)

    # Train RNN
    print('\n=== Training RNN ===')
    rnn = ZebraFinchRNN(N_CLASSES).to(device)
    rnn_history = train_model(rnn, 'rnn', train_loader, val_loader,
                              class_weights_t)

    print('\nTraining complete.')
    print(f'Models saved in: {MODELS_DIR.resolve()}')
