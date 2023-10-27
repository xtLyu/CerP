import cifar100_train
import config


def train(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys, noise_trigger, intinal_trigger):
    epochs_submit_update_dict = {}
    num_samples_dict = {}
    user_grad = []
    server_update = dict()
    if helper.params['type'] == config.TYPE_CIFAR100:
        epochs_submit_update_dict, num_samples_dict, user_grad, server_update, tuned_trigger = cifar100_train.Cifar100Train(
            helper,
            start_epoch,
            local_model,
            target_model,
            is_poison,
            agent_name_keys,
            noise_trigger,
            intinal_trigger)
    return epochs_submit_update_dict, num_samples_dict, user_grad, server_update, tuned_trigger
