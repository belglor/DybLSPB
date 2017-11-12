### DEFINE BATCH LOADER
class batch_loader:
    
    import numpy as np
    
# =============================================================================
#     # Variables
#     batch_size = 0
#     iters_per_epoch = 0
#     epoch_idx = 1
#     data = []
#     labels = []
#     idx = []
#     end_epoch = True
#     
# =============================================================================
    def __init__(self, data, labels, batch_size):  
        import numpy as np
        self.data = data
        self.labels = labels
        self.batch_idx = []
        self.batch_size = batch_size
        self.epoch_idx = 0
        self.iters_per_epoch = len(data)//batch_size
        self.end_epoch = True
        self.idx = np.arange(0, len(data))

    #Methods
    
# =============================================================================
#     def batch_init(data, labels, batch_size):
#         iters_per_epoch = len(data)//batch_size;
#         end_epoch = True;
#         idx =np.arange(0, len(data))
# =============================================================================
            
        
    def next_batch(self):
        import numpy as np
        # If the epoch is finished, reshuffle the indexes and restart epoch
        if(self.end_epoch):  
            np.random.shuffle(self.idx)
            self.end_epoch = False
            self.epoch_idx = 0
        # If we're on our last mini-batch, update flags
        if(self.epoch_idx > self.iters_per_epoch - 1):
            # Not sure if we will have batch_size examples left: just take whats left
            self.batch_idx = self.idx[(self.batch_size)*self.epoch_idx:]
            # print('debug')
            # print(self.batch_idx)
            batch_data = [self.data[i] for i in self.batch_idx]
            batch_labels = [self.labels[i] for i in self.batch_idx]
            self.end_epoch = True
            return np.asarray(batch_data), np.asarray(batch_labels)
        # Take batch_size examples from data
        self.batch_idx = self.idx[(self.batch_size)*self.epoch_idx:self.batch_size*(self.epoch_idx+1)]
        # print('debug')
        # print(self.batch_idx)
        batch_data = [self.data[i] for i in self.batch_idx]
        batch_labels = [self.labels[i] for i in self.batch_idx]
        # Update indexes
        self.epoch_idx = self.epoch_idx + 1;
        return np.asarray(batch_data), np.asarray(batch_labels)
            
    def is_epoch_done(self):
        return self.end_epoch;
    
    def reset_loader(self):
        import numpy as np
        self.batch_idx = [];
        self.epoch_idx = 0;
        self.end_epoch = True;
        self.idx = np.arange(0, len(self.data))

    
    