import os
import torch as th
from torch.utils.data import ConcatDataset
import copy

from .utils import set_initial_seed, string2class
from .serialisation import to_json_file, from_pkl_file, to_torch_file, from_torch_file
from .configurations import create_object_from_config
from .datasets import ListDataset


# class for all experiments
class Experiment:

    def __init__(self, config, output_dir, logger, debug_mode):

        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.debug_mode = debug_mode

        # save config
        to_json_file(self.config, os.path.join(output_dir, 'config.json'))

    ####################################################################################################################
    # DATASET FUNCTIONS
    ####################################################################################################################
    # this methods return trainset, valset
    def __load_training_data__(self):
        dataset_config = self.config.dataset_config

        # todo: concat does not allow on-the-fly loading
        trainset = ConcatDataset(self.__load_list_of_datasets__('train').values())
        valset = ConcatDataset(self.__load_list_of_datasets__('validation').values())

        if 'max_tr_elements' in dataset_config:
            if len(trainset) == 1:
                trainset = trainset[:dataset_config.max_tr_elements]
            else:
                raise ValueError('max_tr_elements options cannnot be set to a list of training data')

        return trainset, valset

    def __load_test_data__(self):
        return self.__load_list_of_datasets__('test')

    def __load_list_of_datasets__(self, tag):
        data_dir = self.config.dataset_config.data_dir

        outdict = {}
        for f in os.listdir(data_dir):
            if tag in f:
                outdict[f] = ListDataset(from_pkl_file(os.path.join(data_dir, f)))
        return outdict

    ####################################################################################################################
    # MODULE FUNCTIONS
    ####################################################################################################################
    def __create_exp_module__(self):
        m = create_object_from_config(self.config.exp_module_config)
        if 'params_path' in self.config.exp_module_config:
            run_dir = os.path.basename(os.path.normpath(self.output_dir))
            params_path = os.path.join(self.config.exp_module_config.params_path, run_dir)
            params_path = os.path.join(params_path, 'params_learned.pth')
            self.logger.warning('Loading model params from {}'.format(params_path))
            params = from_torch_file(params_path)
            m.load_state_dict(params)
        return m

    ####################################################################################################################
    # TRAINER FUNCTIONS
    ####################################################################################################################
    @staticmethod
    def __get_optimiser__(optim_config, model):
        optim_class = string2class(optim_config['class'])
        params_groups = dict(optim_config.params) if 'params' in optim_config else {}
        params_groups.update({'params': [x for x in model.parameters() if x.requires_grad]})

        return optim_class([params_groups])

    def __get_trainer__(self):
        return create_object_from_config(self.config.trainer_config, debug_mode=self.debug_mode, logger=self.logger)

    def __get_training_params__(self, model, device):
        d = copy.deepcopy(self.config.trainer_config.training_params)

        if 'optimiser' in d:
            d['optimiser'] = self.__get_optimiser__(d['optimiser'], model)
        if 'loss_function' in d:
            d['loss_function'] = create_object_from_config(d['loss_function'])
        if 'collate_fun' in d:
            d['collate_fun'] = create_object_from_config(d['collate_fun'], device=device)

        return d

    ####################################################################################################################
    # UTILS FUNCTIONS
    ####################################################################################################################
    def __get_device__(self):
        dev = self.config.others_config.gpu
        cuda = dev >= 0
        device = th.device('cuda:{}'.format(dev)) if cuda else th.device('cpu')
        if cuda:
            th.cuda.set_device(dev)
        return device

    def __save_test_model_params__(self, best_model):
        to_torch_file(best_model.state_dict(), os.path.join(self.output_dir, 'params_learned.pth'))

    ####################################################################################################################
    # TRAINING FUNCTION
    ####################################################################################################################
    def run_training(self, metric_class_list, do_test):
        # initialise random seed
        if 'seed' in self.config.others_config:
            seed = self.config.others_config.seed
        else:
            seed = -1
        seed = set_initial_seed(seed)
        self.logger.info('Seed set to {}.'.format(seed))

        trainset, valset = self.__load_training_data__()

        m = self.__create_exp_module__()
        # save number of parameters
        n_params_dict = {k: v.numel() for k, v in m.state_dict().items()}
        to_json_file(n_params_dict, os.path.join(self.output_dir, 'num_model_parameters.json'))

        dev = self.__get_device__()
        self.logger.info('Device set to {}'.format(dev))
        m.to(dev)

        trainer = self.__get_trainer__()
        training_params = self.__get_training_params__(m, dev)

        # train and validate
        best_val_metrics, best_model, info_training = trainer.train_and_validate(model=m,
                                                                                 trainset=trainset,
                                                                                 valset=valset,
                                                                                 metric_class_list=metric_class_list,
                                                                                 **training_params)

        best_val_metrics_dict = {x.get_name(): x.get_value() for x in best_val_metrics}

        to_json_file(best_val_metrics_dict, os.path.join(self.output_dir, 'best_validation_metrics.json'))
        to_json_file(info_training, os.path.join(self.output_dir, 'info_training.json'))

        if not do_test:
            return best_val_metrics
        else:

            self.__save_test_model_params__(best_model)

            testset_dict = self.__load_test_data__()
            test_metrics_dict = {}
            test_prediction_dict = {}
            avg_metrics = []
            for test_name, testset in testset_dict.items():
                test_metrics, test_prediction = trainer.test(best_model, testset,
                                                             collate_fun=training_params['collate_fun'],
                                                             metric_class_list=metric_class_list,
                                                             batch_size=training_params['batch_size'])
                if len(avg_metrics) == 0:
                    avg_metrics = copy.deepcopy(test_metrics)
                else:
                    for i in range(len(avg_metrics)):
                        avg_metrics[i] = avg_metrics[i] + test_metrics[i]

                test_metrics_dict[test_name] = {x.get_name(): x.get_value() for x in test_metrics}
                test_prediction_dict[test_name] = test_prediction

            to_json_file(test_metrics_dict, os.path.join(self.output_dir, 'test_metrics.json'))
            to_torch_file(test_prediction_dict, os.path.join(self.output_dir, 'test_prediction.pth'))

            # the output is printed. average over all test datasets
            for i in range(len(avg_metrics)):
                avg_metrics[i] = avg_metrics[i]/len(testset_dict)

            return avg_metrics
