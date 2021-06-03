import os
from ComputationalTools import *
import argparse
from sys import exit
import re
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--script_folder", type=str, default="Scripts", help="script_folder")
parser.add_argument("--script_name", type=str, default="", help="script_name")
parser.add_argument("--cluster_name_list", type=str, default="", help="cluster_name_list")
parser.add_argument("--arg_name_list", type=str, default="", help="All names of arguments to be modified")
parser.add_argument("--arg_val_list", type=str, default="", help="All values of arguments to be modified (all arguments in arg_name_list will have the same values).")
parser.add_argument('--use_default_data_folder', action='store_true', help="use_default_data_folder")

args = parser.parse_args()

script_folder = args.script_folder
if args.script_name == "":
    script_name = Get_latest_files("Scripts", delimiter="sh")[0]
else:
    script_name = args.script_name
    if script_name.split(".")[-1] == "sh":
        script_name = ".".join(script_name.split(".")[:-1])

if args.arg_name_list == "":
    raise ValueError("Please specify arg names")
else:
    arg_name_list = [ele for ele in args.arg_name_list.split(",")]

if args.arg_val_list == "":
    raise ValueError("Please specify arg vals")
else:
    if "," in args.arg_val_list:
        delimiter = ","
    else:
        delimiter = "-"

    try:
        arg_val_list = [ast.literal_eval(ele) for ele in args.arg_val_list.split(delimiter)]
    except:
        arg_val_list = [ele for ele in args.arg_val_list.split(delimiter)]

print("script_folder:", script_folder)
print("script_name:", script_name)
print("arg_name_list:", arg_name_list)
print("arg_val_list:", arg_val_list)

# exit()
script_line_dict = {}
script_line_list = []
with open(os.path.join(script_folder, f"{script_name}.sh"), "r") as f:
    script_line_list = f.readlines()
    print("script_line_list", script_line_list)


data_save_path_dict = {"b1": "/mnt/group1/yilong",
                       "b2": "/mnt/group1/yilong",
                       "b3": "/mnt/savefiles/sdb/yilong"}

print("")
original_arg_val = 0
with open(os.path.join(script_folder, f"{script_name}.sh"), "w") as f:
    for k, new_arg_val in enumerate(arg_val_list):
        for j, line in enumerate(script_line_list):
            # print(f"k = {k}, j = {j}, new_arg_val = {new_arg_val}, line = {line}")
            print(f"k = {k}, j = {j}, new_arg_val = {new_arg_val}")
            line_break = line.split(" ")
            new_line = line
            for l, arg_name in enumerate(arg_name_list):
                arg_val = None
                for i, word in enumerate(line_break):
                    if word == f"--{arg_name}":
                        arg_val = line_break[i + 1]
                # print(f"{arg_name}: {new_arg_val}")
                original_arg_val = arg_val
                new_line = new_line.replace(f"--{arg_name} {original_arg_val}", f"--{arg_name} {new_arg_val}")

            f.write(new_line)
            if (not (j == len(script_line_list) - 1 and k == len(arg_val_list) - 1)) and new_line[-1] != "\n":
                """ Add a linebreak if this is the last line """
                # print("Add a linebreak")
                f.write("\n")