from class_resolver import Resolver
from torch.nn import Module

from model.model import BasedModel
from model.lightGCN import LGNModel

from class_resolver import Resolver
from torch.nn import Module

from model.model import BasedModel
from model.lightGCN import LGNModel
from model.bpr import BPRModel
from model.bpr_um import UmBPRModel
from model.lightGCN_um import UmLGNModel
from model.gmf import GMFModel
from model.bpr_sm import SmBPRModel

from torch.optim import Adam, Optimizer
from torch.optim import Adagrad

model_resolver = Resolver(
    {
        BPRModel,
        UmBPRModel,
        SmBPRModel,
        LGNModel,
        UmLGNModel,
        GMFModel,
    },
    base=BasedModel,
    default=Module,
    suffix='model',
)

op_resolver = Resolver(
    {
        Adam,
        Adagrad
    },
    base=Optimizer,
    default=Adagrad,
    # suffix='',
)
