class Config:
    # Training parameters
    EPOCHS = 1000
    BATCH_SIZE = 128
    NUM_WORKERS = 4

      
    
    # Optimizer parameters
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Scheduler parameters
    MAX_LR = 0.01
    PCT_START = 0.2
    DIV_FACTOR = 10
    FINAL_DIV_FACTOR = 100
    
    # Early stopping
    TARGET_ACCURACY = 0.85 

    DROP_OUT = 0.1