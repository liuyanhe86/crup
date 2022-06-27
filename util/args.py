from tap import Tap

class TypedArgumentParser(Tap):
    dataset: str = 'coarse-few-nerd'  # dataset name
    setting: str = 'sup'  # continual learning setting, must be in [sup, CI, online, multi-task]
    model: str = 'Bert-Tagger'  # model name, must be in [PCP, ProtoNet, BERT-Tagger]
    batch_size: int=32  # batch size
    train_epoch: int=10 # num of iters in training
    val_step: int=1  # val after training how many iters
    # warmup_step: int=300  # warm up steps before training
    max_length: int=50 # max length of sentence
    augment: bool=False # whether to use data augment
    lr: float=5e-2  # learning rate for encoder or joint training
    decoder_lr: float=2e-5  # learning rate for classifier training
    load_ckpt: str=None  # load ckpt
    save_ckpt: str=None  # save ckpt
    ckpt_name: str=''  # checkpoint name
    random_seed: int=0  # random seed
    only_test: bool=False  # only test model with checkpoint
    start_task: int=0  # continual task id of beginning task
    pretrain_ckpt: str='bert-base-uncased'  # bert / roberta pre-trained checkpoint
    metric: str='dot'  # metric used to compute distance between embedding and prototypes, must in ['dot', 'L2']
    temperature: float=0.1  # temperature for supervised contrastive loss
    use_sgd: bool=False  # use SGD instead of AdamW for BERT
    only_train_encoder: bool=False  # only train encoder
    only_train_decoder: bool=False  # only train decoder with fixed encoder parameters
    proto_update: str='SDC'  # the way of updating prototypes, only for prototype-based models, must in ['replace', 'mean', 'SDC']
    
    def __str__(self):
        return '; '.join([f'{name}: {self.__getattribute__(name)}' for name in self._get_argument_names()])
    