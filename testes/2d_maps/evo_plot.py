# evaluation libraries

from evo.tools import log, file_interface, plot

log.configure_logging(verbose=False, debug=False, silent=False)
import numpy as np
from evo.core import sync, metrics
from evo.core import trajectory as tra
import math
import matplotlib.pyplot as plt

# file listing
import glob

# table generation
import pandas as pd

# statistics functions
from statistics import stdev, mean

# name of columns for the csv table
table_columns = ['ID',
                 'APE',
                 'RPE-TRANS01',
                 'RPE-TRANS02',
                 'RPE-TRANS03',
                 'RPE-TRANS04',
                 'RPE-TRANS05',
                 'RPE-TRANS06',
                 'RPE-TRANS07',
                 'RPE-TRANS08',
                 'RPE-TRANS09',
                 'RPE-TRANS10',
                 'RPE-ROT01',
                 'RPE-ROT02',
                 'RPE-ROT03',
                 'RPE-ROT04',
                 'RPE-ROT05',
                 'RPE-ROT06',
                 'RPE-ROT07',
                 'RPE-ROT08',
                 'RPE-ROT09',
                 'RPE-ROT10',
                 'KEYPOINTS',
                 'FILTERED KP',
                 'RELOCs',
                 'LOOPS']


def evaluate_sequence(path_to_gt, path_to_sequence):
    # grab the ground truth
    ref_file_path = path_to_gt
    print("\n\nevaluating trajectories in " + path_to_sequence)

    # Means that all files WITH LOG FILES are going to be evaluated
    trajectories = glob.glob(path_to_sequence)
    # continue only if one of more trajectories were found (and store to save
    #                                                     it on the table)
    n_trajs = len(trajectories)
    if n_trajs == 0:
        print("no log found")
        return [], 0

    # list with all results
    results = []

    # calculate APE and RPE for each trajectory
    print(str(n_trajs) + " trajectories found")
    print('evaluating trajectories: ', )
    for t in trajectories:
        # prepare this result
        result = [t[:-4][-16:]]
        # print("-------------Starting to evaluate trajectory ", result, "-------------"
        print('{}... '.format(result))

        # load the ref file
        ref_file = file_interface.read_tum_trajectory_file(ref_file_path)
        print('ref len: {}'.format(ref_file.path_length))
        # store the length of the reference before aligning
        ref_length = ref_file.path_length

        # load the trajectory file
        traj_file = t.replace("log", "trajectory")
        traj_est = file_interface.read_tum_trajectory_file(traj_file)

        # align it with the ground truth
        max_dif = 0.01
        traj_ref, traj_est = sync.associate_trajectories(ref_file,
                                                         traj_est,
                                                         max_dif)
        traj_est.align_trajectory(
            traj_ref,
            correct_scale=True,
            correct_only_scale=False)

        est_aligned = traj_est
        data = (traj_ref, est_aligned)
        print('est len: {}'.format(est_aligned.path_length))

        # compare the length of both trajs
        coverage_rate = est_aligned.path_length / ref_length
        if coverage_rate > 1: coverage_rate = 1
        print('coverage: {}'.format(coverage_rate))
        result.append(coverage_rate)

        # APE
        pose_relation = metrics.PoseRelation.translation_part
        ape_metric = metrics.APE(pose_relation)
        ape_metric.process_data(data)
        ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
        ape = round(ape_stat, 4)
        # print("APE: ", ape

        result.append(ape)

        # RPE

        # calculate delta as 10% of the ground truth length
        # len_ref = (ref_file.path_length) # DEBUG -> trying to
        #                                    # calculate by the original
        #                                    # ground truth, not the
        #                                    # aligned.
        # print("groun truth path length:", len_ref
        # print("start and step: ", (len_ref/10)
        # deltas = np.arange(len_ref/10, len_ref+0.0000001, len_ref/10 )
        deltas = range(100, 801, 100)
        all_pairs = True
        delta_unit = metrics.Unit.meters

        # RPE TRANS
        seq_rpe = []

        pose_relation = metrics.PoseRelation.translation_part
        for delta in deltas:
            # print("Delta ", delta,
            try:
                rpe_metric = metrics.RPE(pose_relation,
                                         delta,
                                         delta_unit,
                                         0.1,
                                         all_pairs)
                rpe_metric.process_data(data)
                stat = rpe_metric.get_statistic(metrics.StatisticsType.mean)
                print("rpe trans stat: ", stat)
                seq_rpe.append(stat / delta)
                # result.append(stat) exchange this
            except Exception as e:
                seq_rpe.append(np.float64(0))
                break
                print('error <---- treat exception on rpe trans')
                # seq_rpe.append(np.float64(99999.9))#exchance this
                # result.append(np.float64(99999.9))# and this
        # print('seq_rpe = {}'.format(seq_rpe)
        rpe_trans = np.sum(seq_rpe) / len(seq_rpe)
        rpe_trans = round(rpe_trans * 100, 4)
        result.append(rpe_trans)

        # print("seq rpe trans: ", seq_rpe

        # RPE ROT
        seq_rpe = []
        pose_relation = metrics.PoseRelation.rotation_angle_deg
        for delta in deltas:
            # print("Delta ", delta,
            try:
                rpe_metric = metrics.RPE(pose_relation,
                                         delta,
                                         delta_unit,
                                         0.1,
                                         all_pairs)
                rpe_metric.process_data(data)
                stat = rpe_metric.get_statistic(metrics.StatisticsType.mean)
                print("rpe rot stat: ", stat)
                seq_rpe.append(stat)
                # result.append(stat)
            except:  # not necessary anymore
                seq_rpe.append(np.float64(0))
                break
                print('error <---- treat exception on rpe rot')
                print(e)
                # append a '_' if there is no trajectory for this delta
                # seq_rpe.append(np.float64(99999.9))
                # result.append(np.float64(99999.9))

        rpe_rot = np.sum(seq_rpe) / len(seq_rpe)
        rpe_rot = round(rpe_rot, 4)
        result.append(rpe_rot)

        # print("seq rpe rot: ", seq_rpe

        # get the number of relocalizations and keypoints filtered
        keypoints = 0  # on this trajectory
        filtered_kp = 0  # on this trajectory

        reloc = -1
        loops = -1
        reloc_line = '-'
        loop_closure_line = '-'
        with open(t, "r") as f:
            for line in f:
                splited_line = line.split('\t')
                if len(splited_line) == 3:
                    keypoints += int(splited_line[1].split(':')[1])
                    filtered_kp += int(splited_line[2].split(':')[1])
                else:
                    if line != '\n':
                        if reloc_line == '-':
                            reloc_line = line
                        else:
                            loop_closure_line = line
            # gambiarrator activate
            reloc = int(reloc_line.split(' ')[-1:][0][:-1])
            if loop_closure_line != '-':
                loops = int(loop_closure_line.split(' ')[-1:][0][:-1])
            else:
                loops = 0
        # append the keypoints, relocalization and loops counts
        result.append(keypoints)
        result.append(filtered_kp)
        result.append(reloc)
        result.append(loops)

        # add this result in the results list
        results.append(result)

        # ploting this trajectory along the groundtruth on the yx plane
        fig = plt.figure()
        traj_by_label = {
            "reference": traj_ref,
            result[0]: est_aligned
        }
        plot.trajectories(fig, traj_by_label, plot.PlotMode.yx)
        plt.savefig(traj_file.replace(".txt", "-yx.png"))

        ## ploting this trajectory along the groundtruth on the zx plane
        # fig = plt.figure()
        # traj_by_label = {
        #    "reference": traj_ref,
        #    result[0]: est_aligned
        # }
        # plot.trajectories(fig, traj_by_label, plot.PlotMode.zx)
        # plt.savefig(traj_file.replace(".txt", "-zx.png"))
        #
        ## ploting this trajectory along the groundtruth on the xyz plane
        # fig = plt.figure()
        # traj_by_label = {
        #    "reference": traj_ref,
        #    result[0]: est_aligned
        # }
        # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        # plt.savefig(traj_file.replace(".txt", "-xyz.png"))

    # create a single list for the mean values of this results
    mean_results = []

    # get the sequence name
    # path = path_to_sequence.split('/')
    # if path[-1] == '': #if it ends with a /
    #    mean_results.append(path[-2])
    # else:              #it the path ends without /
    #    mean_results.append(path[-1])

    # print('len of results col: {}'.format(len(results[0]))

    for column in range(1, len(results[0])):
        # isolate the column
        col = [i[column] for i in results]
        # print('len(col) =  {}'.format(len(col))
        # print('col = {}'.format(col)
        # print('{}: {}'.format(table_columns[column],col)
        # append the mean value
        mean_results.append(round(mean(col), 4))

        # append the std deviation value
        if (len(col) < 2):
            mean_results.append(0)
        else:
            mean_results.append(round(stdev(col), 4))

    # print('len of mean_results: {}'.format(len(mean_results))

    # create the dataframe with all trajectories of this sequence
    # print("\nCreating table for  trajectory ", result, "\n"
    # df = pd.DataFrame(results, columns=table_columns)
    # csv_name = (path_to_sequence).split("/")
    # df.to_csv(path_to_sequence+csv_name[-2]+'.csv',
    #          index=False)
    return mean_results, n_trajs


evaluate_sequence("gt_mh04.csv", "NOVO.txt")
