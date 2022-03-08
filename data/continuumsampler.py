import random

from typing import List, Dict, Tuple

class SampleBase:
    '''
    Abstract Class
    DO NOT USE
    Build your own Sample class and inherit from this class
    '''
    def __init__(self):
        self.class_count = {}

    def get_class_count(self) -> Dict[str, int]:
        '''
        return a dictionary of {class_name : count} in format {any : int}
        '''
        return self.class_count
    
    def valid(self, target_classes: List[str]):
        '''
        decide whether classes of sample is subset of `target_classes`
        '''
        return set(self.get_class_count().keys()).issubset(set(target_classes))


class ContinuumSampler:
    '''
    sample training set and testing set for one episode
    '''
    def __init__(self, N: int, P: int, Q: int, samples: List[SampleBase], classes: List[str]=None, random_state: int=0):
        '''
        `N`: int, how many types in each set
        `P`: int, how many instances for each type in training set
        `Q`: int, how many instances for each type in testing set
        `samples`: List[Sample], Sample class must have `get_class_count` attribute
        `classes`[Optional]: List[any], all unique classes in samples. If not given, the classes will be got from samples.get_class_count()
        `random_state`[Optional]: int, the random seed
        '''
        self.P = P
        self.N = N
        self.Q = Q
        self.samples = samples
        self.__check__() # check if samples have correct types
        if classes:
            self.classes = classes
        else:
            self.classes = self.__get_all_classes__()
        random.seed(random_state)

    def __get_all_classes__(self) -> List[str]:
        '''
        get all classes of samples
        '''
        classes = []
        for sample in self.samples:
            classes += list(sample.get_class_count().keys())
        return list(set(classes))

    def __check__(self) -> None:
        for idx, sample in enumerate(self.samples):
            if not hasattr(sample,'get_class_count'):
                raise ValueError(f'[ERROR] samples in self.samples expected to have `get_class_count` attribute, but self.samples[{idx}] does not')

    def __additem__(self, index: int, set_class: Dict[str, int]) -> None:
        '''
        add class count of `index`-th sample to `set_class`
        '''
        class_count = self.samples[index].get_class_count()
        for class_name in class_count:
            if class_name in set_class:
                set_class[class_name] += class_count[class_name]
            else:
                set_class[class_name] = class_count[class_name]

    def __valid_sample__(self, sample: SampleBase, set_class: Dict[str, int], target_classes: List[str]) -> bool:
        '''
        check whether adding classes of `sample` to `set_class` is valid
        i.e. number of samples per class in `set_class` will in the range of ['P/Q', `2P/2Q`]
        '''
        threshold = 2 * set_class['num_samples_per_class']
        class_count = sample.get_class_count()
        if not class_count:
            return False
        isvalid = False
        for class_name in class_count:
            if class_name not in target_classes:
                return False
            if class_count[class_name] + set_class.get(class_name, 0) > threshold:
                return False
            if set_class.get(class_name, 0) < set_class['num_samples_per_class']:
                isvalid = True
        return isvalid

    def __finish__(self, set_class: Dict[str, int]) -> bool:
        '''
        check whether construction of `set_class` is finished
        i.e. the number of classes in `set_class` == `N` and number of samples per class >= `P`
        '''
        if len(set_class) < self.N + 1:
            return False
        for p in set_class:
            if set_class[p] < set_class['num_samples_per_class']:
                return False
        return True 

    def __get_candidates__(self, target_classes: List[str]) -> List[int]:
        '''
        get indices of samples whose classes is subset of `target_classes`
        '''
        return [idx for idx, sample in enumerate(self.samples) if sample.valid(target_classes)]

    def __next__(self) -> Tuple[List[str], List[int], List[int]]:
        '''
        randomly sample training set and testing set for one episode
        return:
            target_classes: List[any], randomly sampled N classes
            training_idx: List[int], sample index in training set in samples list
            testing_idx: List[int], sample index in testing set in samples list
        '''
        training_class = {'num_samples_per_class':self.P}
        training_idx = []
        testing_class = {'num_samples_per_class':self.Q}
        testing_idx = []
        target_classes = random.sample(self.classes, self.N)  # sampling N classes
        candidates = self.__get_candidates__(target_classes)
        while not candidates:
            target_classes = random.sample(self.classes, self.N)
            candidates = self.__get_candidates__(target_classes)

        # greedy search for training set
        while not self.__finish__(training_class):
            index = random.choice(candidates)
            if index not in training_idx:
                if self.__valid_sample__(self.samples[index], training_class, target_classes):
                    self.__additem__(index, training_class)
                    training_idx.append(index)
        
        # same for testing set
        while not self.__finish__(testing_class):
            index = random.choice(candidates)
            if index not in testing_idx and index not in training_idx:
                if self.__valid_sample__(self.samples[index], testing_class, target_classes):
                    self.__additem__(index, testing_class)
                    testing_idx.append(index)
        
        return target_classes, training_idx, testing_idx

    def __iter__(self):
        return self
    