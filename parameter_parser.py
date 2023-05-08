import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ##################################
    parser.add_argument('--is_vary', type=str2bool, default=False,
                        help='control whether to use multiprocess')
    parser.add_argument('--dataset_name', type=str, default='AIDS',
                        help='options: DD, PROTEINS, DHFR, BZR, COX2, KKI, OHSU,'
                             'ENZYMES, AIDS, FRANKENSTEIN, QM9, NCI1, MUTAG,'
                             '')
    parser.add_argument('--shadow_dataset', type=str, default='AIDS')
    parser.add_argument('--exp', type=str, default='graph_recon',
                        help='options: property_infer, '
                             'subgraph_infer, '
                             'graph_recon, graph_recon_base,'
                             'defense_perturb, ')
    parser.add_argument('--cuda', type=int, default=4,
                        help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)

    ########################## controlling parameters ##############################
    parser.add_argument('--is_split', type=str2bool, default=False)
    parser.add_argument('--is_use_feat', type=str2bool, default=True, help="The impact of node feature.")
    parser.add_argument('--is_train_target_model', type=str2bool, default=False)
    parser.add_argument('--is_use_shadow_model', type=str2bool, default=False)
    parser.add_argument('--is_train_shadow_model', type=str2bool, default=False)
    parser.add_argument('--is_upload', type=str2bool, default=False)
    parser.add_argument('--database_name', type=str, default="subgraph_inference_2_no_feat")
    parser.add_argument('--num_runs', type=int, default=1)

    ########################## target model parameters ###############################
    parser.add_argument('--target_model', type=str, default='mincut_pool',
                        help='options: diff_pool, mincut_pool, mean_pool')
    parser.add_argument('--shadow_model', type=str, default='mincut_pool',
                        help='options: diff_pool, mincut_pool, mean_pool')
    parser.add_argument('--max_nodes', type=int, default=1000)
    parser.add_argument('--target_ratio', type=int, default=0.4)
    parser.add_argument('--shadow_ratio', type=int, default=0.0)
    parser.add_argument('--attack_train_ratio', type=int, default=0.3)
    parser.add_argument('--attack_test_ratio', type=int, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)

    ##################### property inference attack parameters ##########################
    parser.add_argument('--is_gen_embedding', type=str2bool, default=True)

    parser.add_argument('--properties', type=list, default=['num_nodes', 'num_edges', 'density', 'diameter', 'radius'],
                        help='options: num_nodes, num_edges, density, diameter, radius')
    parser.add_argument('--property_num_class', type=int, default=2)

    ###################### subgraph inference attack parameters ##########################
    parser.add_argument('--is_gen_attack_data', type=str2bool, default=True)

    # parser.add_argument('--graph_pooling_method', type=str, default='mincut_pool',
    #                     help='options: diff_pool, mincut_pool, mean_pool')
    parser.add_argument('--train_sample_method', type=str, default='random_walk',
                        help='options: random_walk, snow_ball, forest_fire')
    parser.add_argument('--test_sample_method', type=str, default='random_walk',
                        help='options: random_walk, snow_ball, forest_fire')
    parser.add_argument('--sample_node_ratio', type=float, default=0.8)
    parser.add_argument('--feat_gen_method', type=list, default=['element_l2'],
                        help='options: concatenate, element_l1, element_l2'
                             'cosine_similarity, l2_distance, l1_distance')

    # subgraph inference 1 paras
    parser.add_argument('--attack_model', type=list, default=['mlp'],
                        help='options: mlp, lr, dt')

    # subgraph inference 2 paras

    ####################### graph reconstruct attack parameters ############################
    parser.add_argument('--is_train_gae', type=str2bool, default=False)
    parser.add_argument('--is_use_fine_tune', type=str2bool, default=False)
    parser.add_argument('--is_fine_tune_gae', type=str2bool, default=False)
    parser.add_argument('--is_gen_recon_data', type=str2bool, default=False)
    parser.add_argument('--is_ablation', type=str2bool, default=False)

    parser.add_argument('--gae_num_epochs', type=int, default=100)
    parser.add_argument('--epoch_step', type=int, default=10)
    parser.add_argument('--fine_tune_num_epochs', type=int, default=10)
    parser.add_argument('--encoder_method', type=str, default='diff_pool',
                        help='options: diff_pool, mincut_pool, mean_pool')

    parser.add_argument('--graph_recon_stat', type=list,
                        default=['degree_dist', 'close_central_dist', 'between_central_dist',
                                 'cluster_coeff_dist',
                                 'isomorphism_test'],
                        # default=['degree_dist'],
                        help='options: degree_dist, cluster_coeff_dist, between_central_dist, close_central_dist, '
                             'isomorphism_test')
    parser.add_argument('--graph_recon_metric', type=list,
                        default=['cosine_similarity'],
                        # default=['kl', 'l2', 'wasserstein', 'cosine_similarity'],
                        help='options: l2, wasserstein, cosine_similarity, kl, jsd')

    parser.add_argument('--graph_gen_method', type=str, default='ER',
                        choices=['BA', 'ER'])

    ######################################## defense ###########################################
    parser.add_argument('--noise_std', type=float, default=10.0)
    parser.add_argument('--attack', type=str, default='subgraph_infer_2',
                        choices=['property_infer', 'subgraph_infer_1',
                                  'subgraph_infer_2', 'graph_recon'])

    return vars(parser.parse_args())
