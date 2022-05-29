from tap import Tap

class TypedArgumentParser(Tap):
    dataset: str = 'few-nerd'  # dataset name
    setting: str = 'sup'  # continual learning setting, must be in [sup, CI, online, multi-task]
    model: str = 'Bert-Tagger'  # model name, must be in [PCP, ProtoNet, BERT-Tagger]
    batch_size: int=32  # batch size
    train_epoch: int=10 # num of iters in training
    val_step: int=1  # val after training how many iters
    # warmup_step: int=300  # warm up steps before training
    max_length: int=50 # max length of sentence
    augment: bool=False # whether to use data augment
    encoder_lr: float=2e-5  # encoder learning rate
    classifier_lr: float=1e-3  # classifier learning rate
    load_ckpt: str=None  # load ckpt
    save_ckpt: str=None  # save ckpt
    ckpt_name: str=''  # checkpoint name
    random_seed: int=0  # random seed
    only_test: bool=False  # only test model with checkpoint
    start_task: int=0  # continual task id of beginning task
    pretrain_ckpt: str='bert-base-uncased'  # bert / roberta pre-trained checkpoint
    dot: bool=False  # use dot instead of L2 distance for proto
    temperature: float=0.1  # temperature for supervised contrastive loss
    use_sgd: bool=False  # use SGD instead of AdamW for BERT
    
    def __str__(self):
        return '; '.join([f'{name}: {self.__getattribute__(name)}' for name in self._get_argument_names()])
    