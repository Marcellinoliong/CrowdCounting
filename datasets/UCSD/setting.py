from easydict import EasyDict as edict

# init
__C_UCSD = edict()

cfg_data = __C_UCSD

__C_UCSD.STD_SIZE = (480,720)
__C_UCSD.TRAIN_SIZE = (480,720)
__C_UCSD.DATA_PATH = '/home/marcellino/CrowdCounting/datasets/ProcessedData/UCSD'               

__C_UCSD.MEAN_STD = ([0.376973062754, 0.376973062754, 0.376973062754],[0.206167116761, 0.206167116761, 0.206167116761])

__C_UCSD.LABEL_FACTOR = 1
__C_UCSD.LOG_PARA = 100.

__C_UCSD.RESUME_MODEL = ''#model path
__C_UCSD.TRAIN_BATCH_SIZE = 1 #imgs

__C_UCSD.VAL_BATCH_SIZE = 1 #


