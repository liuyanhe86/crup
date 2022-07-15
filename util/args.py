from tap import Tap

class TypedArgumentParser(Tap):
    dataset: str = 'coarse-few-nerd'  # dataset name
    setting: str = 'sup'  # continual learning setting, must be in [sup, CI, online, multi-task]
    model: str = 'Bert-Tagger'  # model name, must be in [PCP, ProtoNet, BERT-Tagger, AddNER, ExtendNER]
    batch_size: int=32  # batch size
    train_epoch: int=10 # num of iters in training
    val_step: int=1  # val after training how many iters
    # warmup_step: int=300  # warm up steps before training
    max_length: int=50  # max length of sentence
    ignore_index: int=-1  # label without consideration
    augment: str=None # data augmentation, must in [None, 'remove', 'permute']
    lr: float=5e-2  # learning rate
    random_seed: int=0  # random seed
    pretrain_ckpt: str='bert-base-uncased'  # bert / roberta pre-trained checkpoint
    metric: str='dot'  # metric used to compute distance between embedding and prototypes, must in ['dot', 'L2']
    temperature: float=0.1  # temperature for supervised contrastive loss
    optimizer: str='SGD'  
    only_test: bool=False
    start_task: int=0
    proto_update: str='SDC'  # the way of updating prototypes, only for prototype-based models, must in ['replace', 'mean', 'SDC']
    embedding_dimension: int=64  # the dimension of the embedding used for contrastive learning
    alpha: float=0.5
    beta: float=0.5
    gdumb_size: int=1000
    gdumb_check_steps: int=100
    
    def __str__(self):
        return '; '.join([f'{name}: {self.__getattribute__(name)}' for name in self._get_argument_names()])
    
if __name__ == '__main__':
    args = TypedArgumentParser()
    if args.load_ckpt:
        print('1')