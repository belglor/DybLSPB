import numpy as np

### DEFINE BATCH LOADER
class batch_loader:
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
    def __init__(self, data, labels, batch_size, is_balanced, is_fast):
        self.data = data
        self.labels = labels
        self.batch_idx = []
        self.batch_size = batch_size
        self.epoch_idx = 0
        self.iters_per_epoch = len(data)//batch_size
        self.end_epoch = True
        self.idx = np.arange(0, len(data))
        self.is_balanced = is_balanced
        self.is_fast = is_fast
        self.num_batches_processed = 0 
    #Methods
    
# =============================================================================
#     def batch_init(data, labels, batch_size):
#         iters_per_epoch = len(data)//batch_size;
#         end_epoch = True;
#         idx =np.arange(0, len(data))
# =============================================================================
            
        
    def next_batch(self):
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
            batch_data = self.data[self.batch_idx] 
            batch_labels = self.labels[self.batch_idx]
            self.end_epoch = True
        else:
            # Take batch_size examples from data
            self.batch_idx = self.idx[(self.batch_size)*self.epoch_idx:self.batch_size*(self.epoch_idx+1)]
            # print('debug')
            # print(self.batch_idx)
            batch_data = self.data[self.batch_idx]
            batch_labels = self.labels[self.batch_idx]
            # Update indexes
            self.epoch_idx += 1;
        self.num_batches_processed += 1
        # Stratify the folds if stipulated
        if self.is_balanced:
            classes_count = np.sum(np.asarray(batch_labels),0)
            largest_class_count = int(np.max(classes_count))
            i_add = np.zeros(0, int) #empty array
            for i in range(classes_count.shape[0]):
                if classes_count[i] < largest_class_count:
                    class_idx = np.where(self.labels[:,i] == 1)[0]
                    i_add = np.hstack((i_add, np.random.choice(class_idx, size=largest_class_count - int(classes_count[i]), replace=True) ))
            return np.vstack((batch_data, self.data[i_add] )), np.vstack((batch_labels, self.labels[i_add] ))
        return batch_data, batch_labels
            
    def is_epoch_done(self):
        return self.end_epoch or (self.is_fast and self.num_batches_processed >= 2)
    
    def reset_loader(self):
        self.batch_idx = []
        self.epoch_idx = 0
        self.end_epoch = True
        self.idx = np.arange(0, len(self.data))
        self.num_batches_processed = 0

    
    
