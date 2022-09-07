import sys
import subprocess


if __name__ == "__main__":

    # get args as a list
    args = sys.argv[1:]
    list_of_runs = []
    # create multiple runs with 5 different seeds
    for s in [111,222,333,444,555]:
        run = ['python3', './main.py'] + args + ["--seed", str(s)]
        print(run)
        list_of_runs.append(run)


        ps = subprocess.Popen(run)