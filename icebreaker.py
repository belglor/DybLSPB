import os
from scipy.io import loadmat, savemat
#these are the old names. For simplicty, we will use them in the future as well
net_weight_names_PZ = ['conv2d_1_kernel',   
'conv2d_1_bias',
'conv2d_2_kernel',
'conv2d_2_bias',
'dense_1_kernel',
'dense_1_bias',
'dense_2_kernel',
'dense_2_bias',
'output_kernel',
'output_bias' ]
num_PZ_trainables = 10

PHASE1, PHASE2, PHASE3 = 1, 2, 3 

class icebreaker:
    #Phase 0: Ole1
    #Phase 1: Lars1
    #Phase 2: All
    def __init__(self, phase=1, lr=0.005, max_epochs = 100, TEST = False):
        self.TEST = TEST
        self.net_weight_names_DF = None
        word_lr = str(lr)
        word_lr = word_lr[:1] + '-' + word_lr[2:]
        self.name = "_IB_P%d_LR{0}_ME{1}".format(word_lr, max_epochs)  
        self.phase = phase
        result_mat_folder = "./results_mat/"
        self.save_path_perf = result_mat_folder + "performance/"
        self.save_path_numpy_weights = result_mat_folder + "trainedweights/"
        for directory in [self.save_path_perf, self.save_path_numpy_weights]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        if not (self.phase == PHASE3 and self.TEST == True): 
            PZ_toload_weights = "results_mat/trainedweights/piczak_A_unbal_LR0-01_ME300_BAL_WEIGHTS" #the good old file
            print('loading Piczak part from the old days when we trained Piczak alone: ' + PZ_toload_weights)
            self.tw_good_old_PZ = loadmat(PZ_toload_weights)
            
    def manage_weigths(self, archname, net_weight_names_DF):
        self.name = archname + self.name
        self.net_weight_names_DF = net_weight_names_DF
        if self.phase > PHASE1 or self.TEST:
            if self.TEST == False:
                load_phase = self.save_path_numpy_weights + self.name%(self.phase-1)
                print('loading pretrained stuff from last icebreaker phase: ' + load_phase)
            else: 
                load_phase = self.save_path_numpy_weights + self.name%self.phase
                print('test error calculation, loading pretrained stuff from current icebreaker phase: ' + load_phase)
            tw_from_phase = loadmat(load_phase)
            self.pretrained_DF = [tw_from_phase[name] for name in self.net_weight_names_DF]
        # load first layer of PZ
        if self.phase == PHASE3 or (self.phase == PHASE2 and self.TEST): 
            self.pretrained_PZ =  [tw_from_phase   [net_weight_names_PZ[j]] for j in range(2)]
        else:               
            self.pretrained_PZ =  [self.tw_good_old_PZ  [net_weight_names_PZ[j]] for j in range(2)]
        # load all other layers of PZ
        if self.TEST == False: 
            self.pretrained_PZ += [self.tw_good_old_PZ  [net_weight_names_PZ[j]] for j in range(2, num_PZ_trainables)]
        elif self.phase == PHASE3:             
            self.pretrained_PZ += [tw_from_phase   [net_weight_names_PZ[j]] for j in range(2, num_PZ_trainables)]
        
    def shall_DF_be_loaded(self):
        return self.phase > PHASE1 or self.TEST
    
    def good_place_to_store_perf(self):
        return self.save_path_perf + self.name%self.phase
    
    def what_is_trainable(self, AV):
        if self.phase == PHASE1: return AV[0:len(self.net_weight_names_DF)]
        if self.phase == PHASE2: return AV[0:(len(self.net_weight_names_DF)+2)] #DF and first PZ layer will be trained
        if self.phase == PHASE3: return AV #everything will be trained
        
    def save_weights(self, W):
        #we save all that we newly learned (and that can therefore not be found in the old PZ file)
        if self.phase == PHASE1: PZ_W_to_be_saved = []
        if self.phase == PHASE2: PZ_W_to_be_saved = net_weight_names_PZ[:2]
        if self.phase == PHASE3: PZ_W_to_be_saved = net_weight_names_PZ
        savdict = {name : W[k] for k, name in enumerate(self.net_weight_names_DF + PZ_W_to_be_saved)}
        savemat(self.save_path_numpy_weights + self.name%self.phase, savdict)
        

        
        
    
    




    