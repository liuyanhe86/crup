import torch

class Evaluater:

    def __init__(self, ignore_index=-1) -> None:
        self.ignore_index = ignore_index

    def __delete_ignore_index(self, pred, label):
        pred = pred[label != self.ignore_index]
        label = label[label != self.ignore_index]
        assert pred.shape[0] == label.shape[0]
        return pred, label

    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def __transform_label_to_tag__(self, pred, true_label, label2tag):
        '''
        flatten labels and transform them to string tags
        '''
        # drop ignore index
        true_label = true_label[true_label!=self.ignore_index]
        true_label = true_label.cpu().numpy().tolist()
        pred_tag = [label2tag[label] for label in pred]
        label_tag = [label2tag[label] for label in true_label]
        
        assert len(pred_tag) == len(label_tag)
        assert len(pred_tag) == len(pred)
        return pred_tag, label_tag

    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    def __get_type_error__(self, pred, true_label, label2tag):
        '''
        return finegrained type error cnt, coarse type error cnt and total correct span count
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, true_label, label2tag)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span, wrong_outer_span, total_correct_span

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        pred, label = self.__delete_ignore_index(pred, label)
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
                
    def metrics_by_entity(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred = pred.view(-1)
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        pred_class_span = self.__get_class_span_dict__(pred)
        label_class_span = self.__get_class_span_dict__(label)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def error_analysis(self, pred, label, label2tag):
        '''
        return 
        token level false positive rate and false negative rate
        entity level within error and outer error 
        '''
        pred = pred.view(-1)
        true_label = label
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        fp = torch.sum(((pred > 0) & (label == 0)).type(torch.FloatTensor))
        fn = torch.sum(((pred == 0) & (label > 0)).type(torch.FloatTensor))
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        within, outer, total_span = self.__get_type_error__(pred, true_label, label2tag)
        return fp, fn, len(pred), within, outer, total_span