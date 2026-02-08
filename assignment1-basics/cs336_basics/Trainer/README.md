# Trainer

This folder contains the core training utilities used for Transformer-based language model experiments,
including the training loop, loss functions, optimizers, learning rate schedulers, data loading utilities,
and checkpointing support.

---

## Folder Structure

```bash
Trainer/
├── README.md                      # This file
├── __init__.py                    # Module exports
├── trainer.py                     # Main training loop
├── Loss/                          # Loss functions (e.g. cross entropy, perplexity)
│   ├── README.md
│   ├── __init__.py
│   └── cross_entropy_loss.py
├── Optimizer/                     # Optimizers and gradient utilities
│   ├── README.md
│   ├── __init__.py
│   ├── sgd_optimizer.py
│   ├── adamw_optimizer.py
│   └── gradient_clipping.py
├── Scheduler/                     # Learning rate schedulers
│   ├── README.md
│   ├── __init__.py
│   └── cosine_annealing_scheduler.py
├── Data_Loader/                   # Data loading utilities for token streams
│   ├── README.md
│   ├── __init__.py
│   └── data_loader.py
└── Checkpointing/                 # Save/load training checkpoints
    ├── README.md
    ├── __init__.py
    ├── save_checkpoint.py
    └── load_checkpoint.py
```