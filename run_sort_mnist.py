import numpy as np
import time
import copy
import sys
import pathlib
import glob
import os
import subprocess
import re
import tempfile

from sklearn.datasets import make_moons, make_circles, make_classification, load_digits
from sklearn.datasets import fetch_openml
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2

from joblib import Parallel, delayed
import pandas as pd

def get_state(filename, max_lines):
    state = {'pc':[],
             'sp': [],
            'eax':[],
            'ebx':[],
            'ecx':[],
            'edx':[],
            'esi':[]
            }
    nlines = 0
    with open(filename, 'r') as fp:
        for line in fp:
            nlines += 1
            m = re.findall(r"(0x[0-9A-Fa-f]+)", line)
            #print("matches:", m)
            if(m):
                #print("a", m[1], "b", m[2])
                state['pc'].append(int(m[0], 16))
                state['sp'].append(int(m[1], 16))
                state['eax'].append(int(m[2], 16))
                state['ebx'].append(int(m[3], 16))
                state['ecx'].append(int(m[4], 16))
                state['edx'].append(int(m[5], 16))
                state['esi'].append(int(m[6], 16))
            if max_lines and nlines > max_lines:
                print("Got max lines!")
                break

    return state 

def run_case(X, y):
    print(f'Running case {X} label {y}')
    train_sample = X
    #print(train_sample)
    filename = tempfile.NamedTemporaryFile()
    pintool = '/home/lykkebo/pintool/pin-3.16-98275-ge0db48c31-gcc-linux/pin'
    pintool_tool = '/home/lykkebo/pintool/pin-3.16-98275-ge0db48c31-gcc-linux/source/tools/ManualExamples/obj-intel64/proccount_trace.so'
    args = [pintool, '-t', pintool_tool, '-o', filename.name, '--', '/home/lykkebo/stack_reservoir/quicksort'] + list(map(str, train_sample.tolist()))

    subprocess.call(args)
    return get_state(filename.name, 3000), y

def run_circles():
    n_samples = 100
    ntries = 10
    X,y = make_circles(1000)
    #print(X)
    #print(y)
    #quit()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=42)
    para_return = Parallel(n_jobs=-1)(delayed(run_case)(train_case_X, train_case_y)for train_case_X, train_case_y in zip(X_train[:ntries], y_train[:ntries]))
    print(para_return)

def main_worker(args):

    #X, y = fetch_openml('mnist_784', data_home='/home/lykkebo/scikit_learn_data', version=1, return_X_y=True, cache=True)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.01, test_size=0.01, random_state=42)
    #print(f"Testset shape: {X_test.shape}, trainset {X_train.shape} target train {y_train.shape}")
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, train_size=0.5, test_size=0.5, random_state=42)


    genome_file = f"{args.outdir}/genomes.csv"
    print(f'Running with train size {len(X_train)} and testsize {len(X_test)}')
    genome = pd.read_csv(genome_file, header=None).iloc[args.worker_id]
    genome = genome.values.tolist()
    print(f'worker {args.worker_id} read genome: {genome}')

    out_filename = f"{args.outdir}/score_{args.worker_id}"
    #run candidate on many testcases in parallel on node.
    score = run_candidate(genome, X_train, X_test, y_train, y_test)
    #write score to a file for later digging.
    with open(out_filename, 'w') as fp:
        fp.write(str(score) + '\n')


def run_candidate(genome, X_train, X_test, y_train, y_test):

    clf = RidgeClassifier(alpha=1.0)
    para_return = Parallel(n_jobs=-1)(delayed(run_case)(train_case_X, train_case_y) for train_case_X, train_case_y in zip(X_train, y_train))

    states = []
    trains = []
    files = []
  
    for X,y in para_return:
        states.append(X)
        trains.append(y)

    trains = np.asarray(trains, dtype=np.float64)
    #selection_indices = np.asarray(range(1000,1500)) #np.random.random_integers(0, 1000, size=(500,))
    #selection_indices = [i * period for i in range(10)] #np.random.random_integers(0, len(X_train[0]-1), size=(int(len(X_train[0])/2),))
    X_train_pc = [np.asarray(s['sp'])[np.asarray(genome, dtype=np.int)] for s in states]
    print("RESERVOIR:", X_train_pc)
    clf.fit(X_train_pc, trains)

    #TESTING
    states = []
    test_labels = []
    test_para_return = Parallel(n_jobs=-1)(delayed(run_case)(X, y)for X, y in zip(X_test, y_test))

    for X,y in test_para_return:
        states.append(X)
        test_labels.append(y)

    test_labels = np.asarray(test_labels, dtype=np.float64)
    X_test_pc = [np.asarray(s['sp'])[np.asarray(genome, dtype=np.int)] for s in states]

    #for prediction, true_label in zip(clf.predict(X_test_pc), test_labels):
    #    print("Prediction: ", prediction, "True label ", true_label)
    #print("Score:", clf.score(X_test_pc, test_labels))
    return clf.score(X_test_pc, test_labels)

def main_evo(args):
    max_generations = 10
    genome_size = 200
    pc_samples = 3000
    generation_size = 1000
    slurm_template = "runsort.slurm.sh"


    #main evolution loop
    for gen in range(max_generations):
        generation_dir = f"{args.outdir}/gen_{gen}"
        try: 
            pathlib.Path(generation_dir).mkdir(parents=True)
        except FileExistsError:
            print(f"Directory for generation {gen} exists!")
            quit()

        if gen == 0:
            #generate initial genomes
            genomes = pd.DataFrame([np.random.randint(0, pc_samples, size=(genome_size,)) for _ in range(generation_size)])
            genomes.to_csv(f"{args.outdir}/gen_0/genomes.csv", header=False, index=False)
        else:
            #ok so we make some new genomes now.
            print(f"Creating generation {gen}")
            new_generation = []
            scoring = {}
            for scorefile in glob.glob(f'{args.outdir}/gen_{gen-1}/score_*'):
                with open(scorefile) as fp:
                    sc = float(fp.read())
                print("toparse...", scorefile)
                worker_id = scorefile.split('_')
                print(worker_id)
                worker_id = int(worker_id[-1])
                gens = pd.read_csv(f'{args.outdir}/gen_{gen-1}/genomes.csv', header=None).iloc[worker_id]
                scoring[tuple(gens.values.tolist())] = sc

            #take the best one
            best_genome = list(max(scoring, key=scoring.get))
            print("THE ELITE: ", best_genome, scoring[tuple(best_genome)])

            new_generation.append(best_genome)
            #modify a few of the genes...
            mutation_rate = 0.1
            for _ in range(generation_size - 1):
                mutate_index = np.random.randint(0, len(best_genome)-1, size=(int(len(best_genome) * mutation_rate)))
                new_genome = copy.copy(best_genome)
                for i in mutate_index:
                    new_genome[i] = new_genome[i] + np.random.randint(0, 10) #add some random number  to the index
                    new_genome[i] = min(pc_samples - 1, new_genome[i])
                new_generation.append(new_genome)
       
        
            genomes = pd.DataFrame(new_generation)
            genomes.to_csv(f"{generation_dir}/genomes.csv", header=False, index=False)

        #print(scoring)

        with open(slurm_template) as fp:
            tpl = fp.read()
        params = {'basepath': generation_dir}
        script = tpl.format(**params)
        #generate the scripts in generation directory
        with open(f"{generation_dir}/runsort.slurm.sh", 'w') as fp:
            fp.write(script)

        #kick off the array for this generation. Each generation contains
        #many agents, each agent runs many tests in parallell.
        cmd = ['sbatch']
        cmd.append(f'--array=0-{generation_size-1}')
        cmd.append(f'{generation_dir}/runsort.slurm.sh')
        p = subprocess.Popen(cmd)
        p.wait()

        while len(glob.glob(f'{args.outdir}/gen_{gen}/score_*')) != generation_size:
            print("Waiting for score files...")
            print(len(glob.glob(f'{args.outdir}/gen_{gen}/score_*')), glob.glob(f'{args.outdir}/gen_{gen}/score_*'), generation_size)
            time.sleep(2)

        #hang here until we have all scorefiles :(

          

import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', choices=['worker', 'parent'])
    parser.add_argument('--worker-id', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('-o', '--outdir', type=str)

    args = parser.parse_args()

    if args.run == 'worker':
        return main_worker(args)
    return main_evo(args)

if __name__ == '__main__':
    main()
