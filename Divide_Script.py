import os
from ComputationalTools import *
import argparse
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--script_folder", type=str, default="Scripts", help="script_folder")
parser.add_argument("--script_name", type=str, default="", help="script_name")
parser.add_argument("--cluster_name_list", type=str, default="", help="cluster_name_list")
parser.add_argument("--cluster_load_list", type=str, default="", help="cluster_load_list")
parser.add_argument('--use_default_data_folder', action='store_true', help="use_default_data_folder")

args = parser.parse_args()

script_folder = args.script_folder
if args.script_name == "":
    script_name = Get_latest_files("Scripts", delimiter="sh")[0]
else:
    script_name = args.script_name
    if script_name.split(".")[-1] == "sh":
        script_name = ".".join(script_name.split(".")[:-1])

if args.cluster_name_list == "":
    raise ValueError("Please specify cluster name.")
else:
    cluster_name_list = args.cluster_name_list.split(",")

if args.cluster_load_list == "":
    raise ValueError("Please specify cluster load")
else:
    cluster_load_list = [int(ele) for ele in args.cluster_load_list.split(",")]
cluster_load_cumsum_0 = np.cumsum([0] + cluster_load_list)


if len(cluster_name_list) != len(cluster_load_list):
    raise ValueError("Inequal number of cluster names and loads.")

print("script_folder:", script_folder)
print("script_name:", script_name)
print("cluster_name_list:", cluster_name_list)
print("cluster_load_list:", cluster_load_list)


# exit()
script_line_dict = {}
script_line_list = []
with open(os.path.join(script_folder, f"{script_name}.sh"), "r") as f:
    script_line_list = f.readlines()


data_save_path_dict = {"b1": "/mnt/group1/yilong",
                       "b2": "/mnt/group1/yilong",
                       "b3": "/mnt/savefiles/sdb/yilong",
                       "b4": ""}

print("")
for i, cluster_name in enumerate(cluster_name_list):
    filename = f"{script_name}_{cluster_name}.sh"
    print(f"Created {filename}")
    with open(os.path.join(script_folder, filename), "w") as g:
        for line in script_line_list[cluster_load_cumsum_0[i]:cluster_load_cumsum_0[i + 1]]:
            if (not args.use_default_data_folder) and cluster_name[:2] in data_save_path_dict:
                line = f"{line[:-1]} --save_path {data_save_path_dict[cluster_name[:2]]}\n"
            g.write(f"{line}")