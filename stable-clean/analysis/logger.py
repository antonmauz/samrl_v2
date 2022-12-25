USE_TENSORBOARD = False  #: bool
if USE_TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_LOGGER: SummaryWriter

# Everything here is set in runner_jonas.py

LOG_INTERVAL = -1

TRAINING_LOG_STEP = 0
# AUGMENTATION_LOG_STEP = 0
TI_LOG_STEP = 0
# DEBUG_LOG_STEP = 0

USE_WANDB = True  #: bool