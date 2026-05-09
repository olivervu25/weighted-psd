from __future__ import absolute_import, print_function

import argparse
from collections import defaultdict
import os
import sys
# sys.path.append('... working directory to PSD ...')

import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np

import rfsd.experiments.gof_testing_experiments_asymptotic as goft_exp
from rfsd.util import (create_folder_if_not_exist,
                          pretty_file_string_from_dict,
                          nice_str, Timer,
                          store_objects, restore_object)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name',
                        choices=['null', 'null-pilot', 'variance_perturb', 'mean_perturb', 'laplace', 'student-t',
                                 'rbm', 'rbm-25'])
    parser.add_argument('-d', metavar='dim', type=int, nargs='*',
                        default=[1,3,5,7,10,15,20])
    parser.add_argument('-r', '--rounds', type=int, default=0)
    parser.add_argument('-J', '--num-features', metavar='J', type=int,
                        default=10)
    parser.add_argument('-s', '--noise-levels', metavar='sigma', nargs='*',
                        type=float, default=[0,.02,.04,.06])
    parser.add_argument('-o', '--output-dir', default='results')
    parser.add_argument('-t', '--tests', nargs='*', metavar='test')
    parser.add_argument('-g', '--gamma', type=float, default=0.25)
    parser.add_argument('--merge', action='store_true')
    return parser.parse_args()


def experiment_name(args):
    attributes = [('rounds', 'rounds'),
                  ('num_features', 'J'),
                  ('tests', 'tests')]
    if args.gamma != 0.25:
        attributes.append(('gamma', 'gamma'))
    if args.experiment_name == 'rbm':
        attributes.append(('noise_levels', 'noise-levels'))
    # Remove number of dimensions to make file name shorter
    # else:
    #     attributes.append(('d', 'dims'))
    attr_dict = { a[1] : nice_str(getattr(args, a[0])) for a in attributes }
    attr_str = pretty_file_string_from_dict(attr_dict)
    name = '-'.join(['asymptotic_goftest', args.experiment_name, attr_str])
    return name


def run_experiment(args):
    kwargs = {}
    expt_grp_fun = goft_exp.run_goft_experiment_group
    if args.experiment_name == 'null':
        expt_fun = goft_exp.run_gauss_goft_experiment
        kwargs = dict(ds=args.d)
    # Add single dimension variance perturb
    elif  args.experiment_name == 'variance_perturb':
        expt_fun = goft_exp.run_gauss_perturb_goft_experiment
        kwargs = dict(ds=args.d)    
    elif  args.experiment_name == 'mean_perturb':
        expt_fun = goft_exp.run_gauss_mean_goft_experiment
        kwargs = dict(ds=args.d)        
    elif  args.experiment_name == 'laplace':
        expt_fun = goft_exp.run_gauss_laplace_goft_experiment
        kwargs = dict(ds=args.d)
    elif args.experiment_name == 'student-t':
        expt_fun = goft_exp.run_gauss_t_goft_experiment
        kwargs = dict(ds=args.d, df=5, n=2000)
    else:
        expt_fun = goft_exp.run_rbm_fssd_experiment
        expt_grp_fun = goft_exp.run_goft_rbm_experiment_group
        if args.experiment_name == 'rbm':
            kwargs = dict(sigmaPers=args.noise_levels, dx=50, dh=40)
        else:
            kwargs = dict(sigmaPers=args.noise_levels, dx=25, dh=20)

    return expt_grp_fun(args.tests, expt_fun, rounds=args.rounds,
                        plot_results=False, Js=[args.num_features],
                        gamma=args.gamma, **kwargs)


def main():
    sns.set_style('white')
    sns.set_context('notebook', font_scale=3, rc={'lines.linewidth': 3})
    args = parse_arguments()
    output_dir = args.output_dir

    create_folder_if_not_exist(output_dir)
    os.chdir(output_dir)
    print('changed working directory to', output_dir)

    if args.rounds == 0:
        if args.experiment_name in ['null', 'laplace']:
            args.rounds = 500
        elif args.experiment_name == 'student-t':
            args.rounds = 250
        elif args.experiment_name in ['null-pilot', 'rbm']:
            args.rounds = 100 
        else:
            args.rounds = 200
    if args.tests is None:
        # Add Test here
        if args.experiment_name == 'null':
            args.tests = [ 'RFSD',
                          'Gauss FSSD-opt',
                          'PSD r1', 'PSD r2', 'PSD r3','PSD r4']
        elif args.experiment_name == 'null-pilot':          
            args.tests = ['RFSD',
                          'PSD r1', 'PSD r2', 'PSD r3','PSD r4']
        elif args.experiment_name == 'rbm':
            args.tests = [ 'RFSD (RBM)',
                          'Gauss FSSD-opt',
                          
                          'IMQ KSD', 'Gauss KSD',
                          'PSD r1', 'PSD r2', 'PSD r3','PSD r4']
        else:
            args.tests = ['RFSD',
                           'Gauss FSSD-opt',
                        'IMQ KSD', 'Gauss KSD',
                          'PSD r1', 'PSD r2', 'PSD r3','PSD r4']

    if args.merge:
        test_lists = [test_str.split(",") for test_str in args.tests]
        all_tests = [test for tests in test_lists for test in tests]
        args.tests = all_tests

    expt_name = experiment_name(args)
    store_loc = expt_name + '-stored-data'

    #Check if path is valid
    
    path_to_check = store_loc

    if os.path.exists(path_to_check):
        print(f"Path '{path_to_check}' exists.")
    else:
        print(f"Path '{path_to_check}' does not exist")

    if args.merge:
        tests = args.tests
        results = { args.num_features : defaultdict(list) }
        for tests in test_lists:
            args.tests = tests
            test_expt_name = experiment_name(args)
            test_store_loc = test_expt_name + '-stored-data'
            test_results = restore_object(test_store_loc, 'results')
            for J, r in test_results.items():
                results[J].update(r)
            params = restore_object(test_store_loc, 'params')
        store_objects(store_loc, results=results, params=params)
    else:
        try:
            results = restore_object(store_loc, 'results')
            params = restore_object(store_loc, 'params')
            print('reloaded existing data')
        except IOError:
            with Timer('experiment'):
                results, params = run_experiment(args)
            store_objects(store_loc, results=results, params=params)

    ymax = .2 if args.experiment_name == 'null' else 1.05
    # legend_kwargs = dict(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #                      ncol=4, frameon=False)
    goft_exp.show_all_results(results, params, save=expt_name, ymax=ymax,
                              show=False)


if __name__ == '__main__':
    main()
