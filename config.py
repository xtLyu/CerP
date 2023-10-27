import torch
#device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
cuda_idx = torch.cuda.current_device()
device = torch.device("cuda:{}".format(cuda_idx))
AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD ='foolsgold'
AGGR_KRUM = 'krum'
AGGR_TRIMMED_MEAN = 'trimmed_mean'
AGGR_BULYAN = 'bulyan'
AGGR_CRFL='crfl'
AGGR_FEDAVGLR='fedavglr'
AGGR_MEDIAN='median'
AGGR_MKRUM = 'mkrum'
AGGR_FLTRUST='fltrust'
AGGR_FEDLDP='fedldp'
AGGR_FEDCDP='fedcdp'
AGGR_DNC = 'dnc'


ATTACK_LIE='a little'
ATTACK_SYBIL='sybil attack'
ATTACK_SCBA='SCBA'
ATTACK_TAILOR='tailored attacks'

MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
patience_iter=20

TYPE_LOAN='loan'
TYPE_CIFAR100='cifar100'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'
TYPE_DDOS='ddos'
TYPE_REDDIT='reddit'
TYPE_FMNIST='fmnist'