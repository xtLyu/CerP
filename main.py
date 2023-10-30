import argparse
import datetime
import logging
import torch
import train
import test
from cifar100_helper import Cifar100Helper
from utils.utils import dict_html
import utils.csv_record as csv_record
import yaml
import time
import visdom
import numpy as np
import random
import config
import copy
from torch.autograd import Variable

CUDA_DEVICE = 0

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")

vis = visdom.Visdom(port=8098)
criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


def trigger_test_byindex(helper, index, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_index_" + str(index) + "_trigger")


def trigger_test_byname(helper, agent_name_key, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_" + str(agent_name_key) + "_trigger")


def vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0, len(names)):
        _name = names[i]
        _weight = weights[i]
        _is_poison = False
        if _name in adversarial_name_keys:
            _is_poison = True
        helper.target_model.weight_vis(vis=vis, epoch=epoch, weight=_weight, eid=helper.params['environment_name'],
                                       name=_name, is_poisoned=_is_poison)


def vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0, len(names)):
        _name = names[i]
        _alpha = alphas[i]
        _is_poison = False
        if _name in adversarial_name_keys:
            _is_poison = True
        helper.target_model.alpha_vis(vis=vis, epoch=epoch, alpha=_alpha, eid=helper.params['environment_name'],
                                      name=_name, is_poisoned=_is_poison)


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


if __name__ == '__main__':
    print('Start training')
    np.random.seed(1)
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()

    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    if params_loaded['type'] == config.TYPE_CIFAR100:
        helper = Cifar100Helper(current_time=current_time, params=params_loaded,
                                name=params_loaded.get('name', 'cifar100'))
        helper.load_data()
    else:
        helper = None

    logger.info(f'load data done')
    helper.create_model()
    logger.info(f'create model done')
    ### Create models
    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')

    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0

    # load trigger
    if params_loaded['type'] == config.TYPE_CIFAR100:
        data_iterator = helper.test_data
        triggervalue = 1
        for batch_id, (datas, labels) in enumerate(data_iterator):
            x = Variable(cuda(datas, True))
            sz = x.size()[1:]
            intinal_trigger = torch.zeros(sz)
            break
        poison_patterns = []
        for i in range(0, helper.params['trigger_num']):
            poison_patterns = poison_patterns + helper.params[str(i) + '_poison_pattern']
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            intinal_trigger[0][pos[0]][pos[1]] = triggervalue  # +delta i  #？
            intinal_trigger[1][pos[0]][pos[1]] = triggervalue  # +delta i
            intinal_trigger[2][pos[0]][pos[1]] = triggervalue  # +delta i

        noise_trigger = copy.deepcopy(intinal_trigger)

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()

        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  # random choose , maybe don't have advasarial
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                for _name_keys in agent_name_keys:
                    if _name_keys in helper.params['adversary_list']:
                        adversarial_name_keys.append(_name_keys)
            else:  # must have advasarial if this epoch is in their poison epoch
                ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                for idx in range(0, len(helper.params['adversary_list'])):
                    for ongoing_epoch in ongoing_epochs:
                        if ongoing_epoch in helper.params[str(idx) + '_poison_epochs']:
                            if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                                adversarial_name_keys.append(helper.params['adversary_list'][idx])

                nonattacker = []
                for adv in helper.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist + nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary'] == False:
                adversarial_name_keys = copy.deepcopy(helper.params['adversary_list'])
        logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')

        epochs_submit_update_dict, num_samples_dict, user_grad, server_update, tuned_trigger = train.train(
            helper=helper,
            start_epoch=epoch,
            local_model=helper.local_model,
            target_model=helper.target_model,
            is_poison=helper.params['is_poison'],
            agent_name_keys=agent_name_keys,
            noise_trigger=noise_trigger,
            intinal_trigger=intinal_trigger)

        noise_trigger = tuned_trigger

        logger.info(f'time spent on training: {time.time() - t}')
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        is_updated = True

        print(helper.params['aggregation_methods'])
        if helper.params['aggregation_methods'] == config.AGGR_MEAN:
            # Average the models
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'])
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_BULYAN:
            is_updated = helper.bulyan(target_model=helper.target_model, updates=updates,
                                       users_count=int(helper.params['no_models']),
                                       corrupted_count=int(helper.params['thetamin']))

            num_oracle_calls = 1

        elif helper.params['aggregation_methods'] == config.AGGR_MKRUM:
            is_updated = helper.mkrum(target_model=helper.target_model, updates=updates,
                                      corrupted_count=int(helper.params['thetamin']),
                                      users_count=int(helper.params['no_models']))

        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)

            num_oracle_calls = 1


        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)
        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=False, agent_name_key="global")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        if len(csv_record.scale_temp_one_row) > 0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        if helper.params['is_poison']:
            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    noise_trigger=tuned_trigger,
                                                                                    is_poison=True,
                                                                                    visualize=False,
                                                                                    agent_name_key="global")

            csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])

        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        logger.info(f'Done in {time.time() - start_time} sec.')
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)

    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    vis.save([helper.params['environment_name']])
