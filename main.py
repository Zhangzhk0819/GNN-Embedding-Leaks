import logging
import torch
import os

from dispatcher_shell import DispatcherShell
from exp.exp_property_infer import ExpPropertyInfer
from exp.exp_subgraph_infer import ExpSubgraphInfer
from exp.exp_graph_recon import ExpGraphRecon
from exp.exp_graph_recon_base import ExpGraphReconBase
from exp.exp_defense_perturb import ExpDefensePerturb
from exp.exp_defense_adv_train import ExpDefenseAdvTrain
from parameter_parser import parameter_parser
import config


def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handlers
    fh1 = logging.FileHandler(config.LOG_PATH + save_name + '.txt', 'w')
    fh1.setLevel(logging.INFO)
    fh1.setFormatter(formatter)
    logger.addHandler(fh1)


def main(args, exp):
    # config the logger
    logger_name = "_".join((exp, args['dataset_name'], args['target_model'], args['shadow_model'], str(args['cuda'])))
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])

    # subroutine entry for different methods
    if exp == 'property_infer':
        ExpPropertyInfer(args)
    elif exp == 'subgraph_infer':
        ExpSubgraphInfer(args)
    elif exp == 'graph_recon':
        ExpGraphRecon(args)
    elif exp == 'graph_recon_base':
        ExpGraphReconBase(args)
    elif exp == 'defense_perturb':
        ExpDefensePerturb(args)
    elif exp == 'defense_adv_train':
        ExpDefenseAdvTrain(args)
    else:
        raise Exception('unsupported attack')


def dispatcher_shell():
    logging.warning("=" * 40 + "start processes" + "=" * 40)

    # mkl.set_num_threads(args["num_threads"])
    dispatch = DispatcherShell(args)

    if args["exp"] == "property_infer":
        dispatch.property_infer()
    elif args["exp"] == "subgraph_infer_2":
        dispatch.subgraph_infer()
    elif args["exp"] == 'defense_perturb':
        ExpDefensePerturb(args)
    elif args["exp"] == "convergence":
        dispatch.graph_recon()
    elif args["exp"] == "marginal":
        dispatch.defense_perturb()
    elif args["exp"] == 'graph_recon':
        dispatch.graph_recon()
    else:
        raise Exception('invalid experiment name')

    if len(args) != len(dispatch.input_args):
        wrong_paras = set(args) - set(dispatch.input_args)
        raise Exception('wrong para names: %s, please check' % (wrong_paras,))

    logging.warning("=" * 40 + "processes finished" + "=" * 40 + "\n" * 3)


if __name__ == "__main__":
    args = parameter_parser()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda'])

    if args['is_vary']:
        dispatcher_shell()
        main(args, args['exp'])
    else:
        main(args, args['exp'])
